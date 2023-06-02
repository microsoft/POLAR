import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support

from nlp_models import get_token_ids, get_entity_masks, LR, MLP, BERTclf, BERTre

from tqdm import tqdm


"""
Training harmonizer using Pareto optimization by minimizing the scalarized loss.
"""


class MultiObjectiveModel:
    
    def __init__(self, model_type, loss_aggr, input_size=None, n_class=2, 
                 learning_rate=1e-5, weight_decay=1e-5, batch_size=16,
                 bert_model="bert-base-uncased") -> None:
        """
        Class for performing Pareto optimization.
        model_type: NLP model type from nlp_models
        loss_aggr: loss aggregation function. Options: linear, square, 2-norm, max
        """
        
        self.model_type = model_type
        self.loss_aggr = loss_aggr
        self.n_class = n_class
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        
        self.input_size = input_size
        self.text_feature = input_size is None
        self.bert_model = bert_model
        
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        torch.cuda.empty_cache()
        
        # initialize model
        if self.model_type == 'LR':
            self.model = LR(input_size=self.input_size, n_class=self.n_class).to(self.device)
        elif self.model_type == 'MLP':
            self.model = MLP(input_size=self.input_size, n_class=self.n_class).to(self.device)
        elif self.model_type == 'BERTclf':
            self.model = BERTclf(n_class=self.n_class, bert_model=self.bert_model).to(self.device)
        elif self.model_type == 'BERTre':
            self.model = BERTre(n_class=self.n_class, bert_model=self.bert_model).to(self.device)
            
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.n_class <= 2:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            
            
    
    def fit(self, X, Y, Xval, Yval, weights=None):
        """
        Fit harmonizer using Pareto optimization.
        X: training input
        Y: training weak labels from the multiple sources. No gold label used.
        Xval: held out validation set input for monitoring the training epoches.
        Yval: validation labels
        """
        Y = torch.as_tensor(Y)
        if len(Y.shape) == 1:
            Y = Y.reshape((len(Y), 1))
            
        if weights:
            self.weights = weights
        else:
            # equal weights on all sources
            self.weights = [1 / Y.shape[1] for _ in range(Y.shape[1])]
        
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(100):
            
            # do one epoch of model training
            self.train(X, Y, self.batch_size)
            
            # Validate the model on the validation set
            val_loss = self.validate(Xval, Yval, batch_size=16)

            # Check if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            print(val_loss, end=' ')

            # Stop training if the validation loss hasn't improved for `patience` epochs
            if counter >= patience:
                print(f"Validation loss hasn't improved for {patience} epochs, stopping training at epoch {epoch}...")
                break
        #self.model.to(torch.device('cpu'))
            
            
            
    def train(self, X, Y, batch_size):
        """
        Training one epoch through the training set X, Y (weak labels)
        """
        n = len(X)
        m = Y.shape[1]
        
        perm = np.random.permutation(n)
        
        #abstain_loss = loss_fn(torch.zeros(1), torch.zeros(1)).item()
        #margin_loss = (loss_fn(torch.ones(1), torch.ones(1)).item() + loss_fn(-torch.ones(1), torch.ones(1)).item()) / 2
        
        self.model.train()
        model_device = next(self.model.parameters()).device
        
        n_batches = int((n-1) / batch_size) + 1
        for batch in tqdm(range(n_batches), disable='BERT' not in self.model_type):
            batch_examples = [X[_] for _ in perm[batch*batch_size : (batch+1)*batch_size]]
            
            output = self.model_output_for_batch(batch_examples)
            
            Y_batch = Y[perm[batch*batch_size : (batch+1)*batch_size], :].to(model_device)
            
            if m == 1:
                # Y is a single label, compute loss directly
                nonabs = Y_batch[:,0] >= 0
                if self.n_class <= 2:
                    loss = self.criterion(output[nonabs,:], Y_batch[nonabs,:].float()).sum()
                else:
                    loss = self.criterion(output[nonabs,:], Y_batch[nonabs,0]).sum()
                
            else:
                # Y is weak label vector, do multi-objective loss aggregation
                batch_losses = self.compute_losses(output=output.float(), weak_labels=Y_batch)
                
                if len(batch_losses) == 0:
                    continue
                
                nb = len(batch_losses)
                
                if self.loss_aggr == 'linear':
                    loss = sum([sum([losses[j]*self.weights[j] for j in losses]) for losses in batch_losses]) / nb
                    
                elif self.loss_aggr == 'square':
                    loss = sum([torch.square(sum([losses[j]*self.weights[j] for j in losses])) for losses in batch_losses]) / nb
                    
                elif self.loss_aggr == '2-norm':
                    loss = sum([torch.norm(torch.stack([losses[j]*self.weights[j] for j in losses]), dim=0) for losses in batch_losses]) / nb
                        
                elif self.loss_aggr == 'max':
                    loss = sum([max([losses[j]*self.weights[j] for j in losses]) for losses in batch_losses]) / nb
                
            # Backpropagationn
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

    
    def validate(self, Xval, Yval, batch_size):
        """Compute loss on validation set."""
        n = len(Xval)
        
        self.model.eval()
        model_device = next(self.model.parameters()).device
        
        running_loss = 0.0
        with torch.no_grad():
            n_batches = int((n-1) / batch_size) + 1
            for batch in range(n_batches):
                batch_examples = Xval[batch*batch_size : (batch+1)*batch_size]
                output = self.model_output_for_batch(batch_examples)
                    
                Y_batch = torch.as_tensor(Yval[batch*batch_size : (batch+1)*batch_size]).to(model_device)
                    
                if self.n_class <= 2:
                    loss = self.criterion(output[:,0].float(), Y_batch.float()).sum()
                else:
                    loss = self.criterion(output, Y_batch).sum()
                    
                running_loss += loss.item()
        return running_loss / n
    
    
    def model_output_for_batch(self, batch_examples):
        model_device = next(self.model.parameters()).device
        
        if not self.text_feature:
            input = torch.Tensor(batch_examples).to(model_device)
            output = self.model(input.float())
            
        elif self.model_type == 'BERTclf':
            input_ids = get_token_ids(batch_examples, self.bert_model)
            input_ids = torch.as_tensor(input_ids).to(model_device)
            output = self.model(input_ids)
            
        elif self.model_type == 'BERTre':
            input_ids, e1_mask, e2_mask = get_entity_masks(batch_examples, self.bert_model)
            input_ids = torch.as_tensor(input_ids).to(model_device)
            e1_mask = torch.as_tensor(e1_mask).to(model_device)
            e2_mask = torch.as_tensor(e2_mask).to(model_device)
            output = self.model(input_ids, e1_mask, e2_mask)
        
        return output
    
    
    def compute_losses(self, output, weak_labels):
        """
        Compute losses on the weak labels in batch. Each voting weak source gives a loss.
        """
        m = weak_labels.shape[1]
        batch_losses = []
        for i in range(len(output)):
            losses = {}
            for j in range(m):
                if weak_labels[i,j].item() != -1:
                    target = weak_labels[i:i+1,j:j+1]
                    if self.n_class <= 2:
                        losses[j] = self.criterion(output[i:i+1,:], target.float())
                    else:
                        losses[j] = self.criterion(output[i:i+1,:], target[:,0])
            if len(losses) == 0:
                continue
            batch_losses.append(losses)
            
        return batch_losses
    
    
    def predict_proba(self, X):
        n = len(X)
        batch_size = 16
        self.model.eval()
        
        probas = []
        with torch.no_grad():
            n_batches = int((n-1) / batch_size) + 1
            for batch in tqdm(range(n_batches)):
                batch_examples = X[batch*batch_size : (batch+1)*batch_size]
                output = self.model_output_for_batch(batch_examples)
                if self.n_class <= 2:
                    probas.append(torch.sigmoid(output).detach().cpu())
                else:
                    probas.append(nn.Softmax(dim=1)(output).detach().cpu())
        return torch.cat(probas, dim=0).squeeze().numpy()
    
        
    def predict(self, X):
        p = self.predict_proba(X)
        if self.n_class <= 2:
            return 1 * (p >= 0.5)
        else:
            return np.argmax(p, axis=1)
    
    
    def score(self, X, Y):
        Y = np.array(Y)
        Ypred = self.predict(X)
        return (Y==Ypred).sum() / len(Y)
    
    
    def evaluate(self, Xtest, Ytest):
        acc = self.score(Xtest, Ytest)
        Ypred = self.predict(Xtest)
        
        precision = precision_recall_fscore_support(Ytest, Ypred)[0][1]
        recall = precision_recall_fscore_support(Ytest, Ypred)[1][1]
        f1 = precision_recall_fscore_support(Ytest, Ypred)[2][1]

        return acc, f1, precision, recall
   