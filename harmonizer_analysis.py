import numpy as np
from sklearn.metrics import precision_recall_fscore_support



class Harmonizer:
    """
    Harmonizer TRAINED on all supervision sources already. Use distribution prediction on
    the train/valid/test examples to perform further analysis, including harmonizer prediction,
    POLAR score computing, calibration evaluation, and source reweighting.
    """
    
    def __init__(self, probas) -> None:
        """
        Initialize from predicted probabilities from the harmonizer. Everything else to be derived upon this.
        """
        self.probas = np.array(probas)
        if len(self.probas.shape) == 1:
            self.probas = self.probas.reshape((len(self.probas), 1))
        self.n, self.n_class = self.probas.shape
        

        
    def predict(self):
        if self.n_class == 1:
            return 1 * (self.probas[:,0] >= 0.5)
        else:
            return np.argmax(self.probas, axis=1)
        
        
    def evaluation_results(self, target):
        pred = self.predict()
        acc = (pred == np.array(target)).sum() / self.n
        
        precision = precision_recall_fscore_support(target, pred, warn_for=())[0][0]
        recall = precision_recall_fscore_support(target, pred, warn_for=())[1][0]
        f1 = precision_recall_fscore_support(target, pred, warn_for=())[2][0]

        return acc, f1, precision, recall
    
    
    def compute_losses(self, weak_labels):
        losses = 1.0 * np.zeros_like(weak_labels)
        
        for i in range(losses.shape[0]):
            for j in range(losses.shape[1]):
                if weak_labels[i][j] < 0:
                    continue
                l = weak_labels[i][j]
                if self.n_class == 1:
                    losses[i,j] = -np.log(self.probas[i,0] if l == 1 else (1 - self.probas[i,0]))
                else:
                    losses[i,j] = -np.log(self.probas[i,l])
        return losses
    
    
    
    
    def compute_loss_gradient(self, weak_labels):
        L = np.array(weak_labels)
        if len(L.shape) == 1:
            L = L.reshape((len(L), 1))
        mask = L >= 0
        
        if self.n_class == 1:
            grad = L - self.probas
            grad = mask * grad
        else:
            grad = np.zeros((L.shape[0], L.shape[1], self.n_class))
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    if L[i,j] < 0:
                        continue
                    grad[i, j, int(L[i,j])] = 1.0
                    grad[i, j, :] = grad[i, j, :] - self.probas[i, :]

        return grad
    
    
    def gradient_correlation(self, weak_labels):
        m = len(weak_labels[0])
        grad = self.compute_loss_gradient(weak_labels)
        
        if self.n_class == 1:
            return np.corrcoef(grad.T)
        
        else:
            Cov = np.zeros((m,m))
            for j1 in range(m):
                for j2 in range(j1, m):
                    g1 = grad[:,j1,:] - grad[:,j1,:].mean(axis=0, keepdims=True)
                    g2 = grad[:,j2,:] - grad[:,j2,:].mean(axis=0, keepdims=True)
                    c = (g1 * g2).sum(axis=1).mean()
                    if j1==j2 and c<= 0:
                        print(j1, j2)
                    Cov[j1,j2] = c
                    Cov[j2,j1] = c
            Corr = np.zeros((m,m))
            for j1 in range(m):
                for j2 in range(m):
                    Corr[j1,j2] = Cov[j1,j2] / np.sqrt(Cov[j1,j1]*Cov[j2,j2])
            return Corr
    
    
    def max_eigenvalue_reweighting(self, weak_labels, min_ratio=0.5):
        Corr = self.gradient_correlation(weak_labels)
        m = len(weak_labels[0])
        
        w = np.ones((m, 1))
        w_min = min_ratio / m    # at least some portion of equal weighted value
        
        for iter in range(10):
            w = Corr @ w    # power iteration
            
        w = w / np.sum(w)    # normalize weight vector
        w_e = w - w_min    # excess weight from minimal value
        w_e = np.maximum(0, w_e)    # all excess weights should be positive
        w_e = w_e / np.sum(w_e) * (1-min_ratio)    # normalize excess weights to be the other portion
        w = w_e + w_min
        
        return list(w[:,0])
    
    
    def min_variance_reweighting(self, weak_labels, min_ratio=0.5):
        Corr = self.gradient_correlation(weak_labels)
        m = len(weak_labels[0])
        
        Weight = np.linalg.inv(Corr) @ np.ones((m, 1))
        
        w = Weight / np.sum(Weight)
        
        w_min = min_ratio / m    # at least half of equal weighted value
        
        w_e = w - w_min    # excess weight from minimal value
        w_e = np.maximum(0, w_e)    # all excess weights should be positive
        w_e = w_e / np.sum(w_e) * (1-min_ratio)    # normalize excess weights to be the other half
        w = w_e + w_min
        
        return list(w[:,0])
    
    
    
    def rank_doubt_scores(self, weak_labels):
        """
        Compute the doubt score (POLAR score) on all weak sources for all examples where the source voted.
        The examples are ranked from high doubt to low doubt for each source j
        """
        pred = self.predict()
        
        m = len(weak_labels[0])
        grad = self.compute_loss_gradient(weak_labels)
        
        doubt_scores = {j:[] for j in range(m)}
        for j in range(m):
            for i in range(self.n):
                if self.n_class == 1:
                    d = np.abs(grad[i,j])
                else:
                    d = grad[i,j, weak_labels[i][j]]
                    
                if d >= 1e-8:
                    doubt_scores[j].append((d, i, pred[i]))
                    # record example as: doubt score, ID, predicted label
            doubt_scores[j] = sorted(doubt_scores[j], key=lambda _: _[0], reverse=True)
            
        return doubt_scores
    
    
    def sample_auto_examples(self, weak_labels, k=2, temperature=0.1, seed=100):
        """
        automatically sample few-shot examples based on the POLAR scores
        """
        np.random.seed(seed)
        doubt_scores = self.rank_doubt_scores(weak_labels)
        m = len(doubt_scores)
        n = len(doubt_scores[m-1])
        j = m - 1
        sampled_examples = []    # doubt_score, example ID, predicted label
        if self.n_class == 1:
            classes = [0, 1]
        else:
            classes = list(range(self.n_class))
            
        for c in classes:
            class_mask = 1 * (np.array([d[2] for d in doubt_scores[j]]) == c)
            # hard examples
            sample_proba = np.exp(np.array([d[0] for d in doubt_scores[j]]) / temperature)
            sample_proba = class_mask * sample_proba
            sample_proba = sample_proba / sample_proba.sum()
            sample_index = np.random.choice(n, size=k, replace=False, p=sample_proba)
            sampled_examples += [doubt_scores[j][i] for i in sample_index]
            # confirmed examples
            sample_proba = np.exp(np.array([1-d[0] for d in doubt_scores[j]]) / temperature)
            sample_proba = class_mask * sample_proba
            sample_proba = sample_proba / sample_proba.sum()
            sample_index = np.random.choice(n, size=k, replace=False, p=sample_proba)
            sampled_examples += [doubt_scores[j][i] for i in sample_index]
            
        return sampled_examples
                
            

    def calibration_bar(self, weak_labels, Y):
        """
        Return outputs for calibration bar plot and compute ECE
        """
        doubt_scores = self.rank_doubt_scores(weak_labels)
        # The harmonizer is used to check against the weak labels and assign doubt score for each source j on each voted example.

        j = len(doubt_scores) - 1
        n = len(doubt_scores[j])

        errors = {bin:[] for bin in range(10)}
        risks = {bin:[] for bin in range(10)}
        for i in range(int(n)):
            ex = doubt_scores[j][i]
            s = ex[0]
            bin = min(int(s*10), 9)
            ii = ex[1]
            errors[bin].append(int(weak_labels[ii][j] != Y[ii]))
            risks[bin].append(s)

        empty = [bin for bin in errors if len(errors[bin])==0]
        for bin in empty:
            errors.pop(bin)
            risks.pop(bin)

        mean_error = [np.mean(errors[bin]) for bin in errors]
        mean_risk = [np.mean(risks[bin]) for bin in risks]
        bin_sizes = [len(errors[bin]) for bin in errors]
        std_error = [np.std(errors[bin])/np.sqrt(len(errors[bin])) for bin in errors]
        ECE = (np.abs(np.array(mean_error) - np.array(mean_risk)) * np.array(bin_sizes)).sum() / sum(bin_sizes)

        return errors, mean_error, std_error, ECE


    def correlation_scatter(self, weak_labels, Y, new_LLM_labels=None):
        """
        Return outputs for calibration scatter plot and compute R2
        new_LLM_labels is optional to take updated LLM labels from dynamic prompting, giving comparing scatter plots
        """
        doubt_scores = self.rank_doubt_scores(weak_labels)
        j = len(doubt_scores) - 1
        n = len(doubt_scores[j])

        error_rate = []
        h_error_rate = []
        top_L = []
        top_L_new = []
        top_Y = []
        top_P = []
        top_doubt = []
        for i in range(int(n)):
            ex = doubt_scores[j][i]
            ii = ex[1]
            top_L.append(weak_labels[ii][j])
            if new_LLM_labels:
                top_L_new.append(new_LLM_labels[ii][0])
            top_Y.append(Y[ii])
            top_P.append(ex[2])
            top_doubt.append(ex[0])
            error_rate.append((np.array(top_L)!=np.array(top_Y)).sum() / len(top_L))
            h_error_rate.append((np.array(top_P)!=np.array(top_Y)).sum() / len(top_L))

        n = len(top_Y)
        bin_size = int(np.sqrt(n))
        n_bins = n // bin_size

        error_rate_bin = []
        h_error_rate_bin = []
        doubt_score_bin = []
        error_rate_bin_new = []
        for b in range(n_bins):
            error_rate_bin.append((np.array(top_L[b*bin_size:(b+1)*bin_size])!=np.array(top_Y[b*bin_size:(b+1)*bin_size])).sum() 
                            / len(top_Y[b*bin_size:(b+1)*bin_size]))
            if new_LLM_labels:
                error_rate_bin_new.append((np.array(top_L_new[b*bin_size:(b+1)*bin_size])!=np.array(top_Y[b*bin_size:(b+1)*bin_size])).sum() 
                            / len(top_Y[b*bin_size:(b+1)*bin_size]))
            h_error_rate_bin.append((np.array(top_P[b*bin_size:(b+1)*bin_size])!=np.array(top_Y[b*bin_size:(b+1)*bin_size])).sum() 
                            / len(top_Y[b*bin_size:(b+1)*bin_size]))
            doubt_score_bin.append(np.mean(top_doubt[b*bin_size:(b+1)*bin_size]))

        return error_rate_bin, doubt_score_bin, error_rate_bin_new