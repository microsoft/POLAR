from pareto_learning import MultiObjectiveModel
from utils import get_feature
import os, json, glob
import pandas as pd
import numpy as np


OpenAI_API_Config = {
                    'api_key': 'REPLACE WITH YOUR API KEY',
                     'api_base': 'REPLACE WITH YOUR API ENDPOINT',
                     'api_type': 'azure',    #OPTIONALLY REPLACE WITH YOUR API TYPE
                     'api_version': "2023-03-15-preview"    #OPTIONALLY REPLACE WITH YOUR API VERSION
                     }



relation_data = ['chemprot', 'cdr', 'semeval']
biomed_data = ['chemprot', 'cdr']

LearnRates = {'-4': 1e-4, '-5': 1e-5, '-6': 1e-6}
WeightDecays = {'-4': 1e-4, '-5': 1e-5}


PATH = os.path.dirname(__file__)


"""
Functions to run the experiments in the paper.
"""


def train_harmonizer(name, gpt_model, model_type, loss_aggr, lr, wd, weights=None, version=''):
    """
    Train harmonizer for specific configuration. Save harmonizer output (class probabilities) to file.
    """
    filepath = os.path.join(PATH, f'/Model_output{version}', name)
    result_name = '_'.join([gpt_model, model_type, loss_aggr, lr, wd]) +'.json'
    if os.path.exists(os.path.join(filepath, result_name)):
        return
    
    if model_type in ['LR', 'MLP']:
        raw_text_feature = False
    else:
        raw_text_feature = True
        model_type += 're' if name in relation_data else 'clf'
    bert_model = "bert-base-cased" if name not in biomed_data else "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    
    X, Y, L, L_llm = get_feature(name, raw_text_feature=raw_text_feature, gpt_model=gpt_model, new_gpt_query=False)
    input_size = None if raw_text_feature else len(X['train'][0])
    n_class = max(Y['train']) + 1

    model = MultiObjectiveModel(model_type=model_type, loss_aggr=loss_aggr, input_size=input_size, n_class=n_class,
                                learning_rate=LearnRates[lr], weight_decay=WeightDecays[wd], bert_model=bert_model)
    
    # add LLM label
    for fold in L:
        for i in range(len(L[fold])):
            L[fold][i].append(L_llm[fold][i][0])
    
    model.fit(X['train'], L['train'], X['valid'], Y['valid'], weights=weights)
    
    Proba = {}
    for fold in X:
        Proba[fold] = model.predict_proba(X[fold]).astype('float').tolist()
        
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(os.path.join(filepath, result_name), "w") as outfile:
        json.dump(Proba, outfile)
        

def run_exp(name, gpt_model, weights=None, version=''):
    """
    Run harmonizer training experiments through all configurations.
    """
    for model_type in ['LR', 'MLP', 'BERT']:
        for loss_aggr in ['linear', 'square', '2-norm', 'max']:
            for lr in LearnRates:
                for wd in WeightDecays:
                    train_harmonizer(name, gpt_model, model_type, loss_aggr, lr, wd, weights=weights, version=version)
                        
                        

def match_harmonizer(name, gpt_model, fold='train', version='', model_type='', loss_aggr='', metric='Loss'):
    """
    Get the harmonizer for specific model type and loss aggregation function. 
    """
    X, Y, L, L_llm = get_feature(name, raw_text_feature=True, gpt_model=gpt_model, new_gpt_query=False)
    # combine LLM label with weak labels
    weak_labels = [L[fold][i] + [L_llm[fold][i][0]] for i in range(len(L[fold]))]
    
    Results = pd.DataFrame(columns=['Accuracy', 'F-1', 'Loss'])
    if 'LLMs' not in version:
        file_format = os.path.join(PATH, f'Model_output{version}/{name}/{gpt_model}_*.json')
    else:
        file_format = os.path.join(PATH, f'Model_output{version}/{name}/*.json')
    filelist = glob.glob(file_format)
    for file in filelist:
        if model_type not in file or loss_aggr not in file:
            continue
        prefix_len = len(file_format) - 6
        spec = file[prefix_len:-5]
        outputs = json.load(open(file))
        h = Harmonizer(outputs['valid'])
        if h.probas.var(axis=0).min() < 1e-8:
            continue
        # Loss
        loss = np.mean(h.compute_losses([[y] for y in Y['valid']]))
        # Accuracy and F-1
        acc, f1, p, r = h.evaluation_results(Y['valid'])
        
        Results.loc[spec] = (acc, f1, loss)

    # optimize configuration on validation set
    if metric in ['Loss']:
        best = Results.index[Results[metric].argmin()]
    else:
        best = Results.index[Results[metric].argmax()]
    file = f'{file_format[:-6]}{best}.json'
    outputs = json.load(open(file))
    print(best)
    
    return X, Y, weak_labels, Harmonizer(outputs[fold])
                        
                        
                        
def reweighted_training(name, gpt_model, method='MaxEig'):
    """
    Train harmonizer using source reweighting techniques.
    """
    X, Y, weak_labels, harmonizer = match_harmonizer(name, gpt_model, fold='train')
    
    if method == 'MaxEig':
        weights = harmonizer.max_eigenvalue_reweighting(weak_labels, min_ratio=0.5)
    elif method == 'MinVar':
        weights = harmonizer.min_variance_reweighting(weak_labels, min_ratio=0.5)
        
    run_exp(name, gpt_model, weights=weights, version=method)
                            
                            
                            
                            

from harmonizer_analysis import Harmonizer
from rule_explanations import get_explanation_from_lfs
                    
def auto_few_shot(name, gpt_model, k=2, explain=False, random=False):
    """
    Auto few-shot based on POLAR scores
    """
    X, Y, weak_labels, harmonizer = match_harmonizer(name, gpt_model)
    labels = json.load(open(os.path.join(PATH, 'wrench', name, 'label.json')))
    
    if not random:
        samples = harmonizer.sample_auto_examples(weak_labels, k=k, temperature=0.01, seed=100)
        version = 'auto_few-shot'
        if explain:
            version = version + '_explain'
    else:
        # randomly sample and take zero-shot label as output
        random_draw = np.random.choice(len(weak_labels), size=k*len(labels)*2, replace=False)
        samples = [[0, i, weak_labels[i][-1]] for i in random_draw if weak_labels[i][-1]>=0]
        version = 'random_few-shot'
    
    auto_examples = []
    for ex in samples:
        ii = ex[1]
        x = X['train'][ii]
        pred = ex[2]
        pred_label = labels[str(pred)]
        if name in relation_data:
            input = ' '.join([x['text'], f"Entity 1: {x['entity1']}", f"Entity 2: {x['entity2']}"])
        else:
            input = x
        if explain:
            evidences = get_explanation_from_lfs(x, weak_labels[ii][:-1])
            pred_label = pred_label + '. \n' + evidences
        auto_examples.append((input, pred_label))
    
    results_exist = os.path.exists(os.path.join(PATH, f'LLM_labels/{gpt_model}/{version}/', name))
    query_gpt = not results_exist
    fix = results_exist
    X, Y, L, L_llm = get_feature(name, raw_text_feature=True, folds=['test'], gpt_model=gpt_model, 
                                 version=version, examples=auto_examples, query_gpt=query_gpt, fix=fix)
    
    

def harmonize_LLMs(name, gpt_models, model_type, loss_aggr, lr, wd, weights=None):
    """
    Train harmonizer only on the LLM labels
    """
    filepath = os.path.join(PATH, f'Model_output_LLMs/{"+".join(gpt_models)}/{name}/')
    result_name = '_'.join([model_type, loss_aggr, lr, wd]) +'.json'
    if os.path.exists(os.path.join(filepath, result_name)):
        return
    
    if model_type in ['LR', 'MLP']:
        raw_text_feature = False
    else:
        raw_text_feature = True
        model_type += 're' if name in relation_data else 'clf'
    bert_model = "bert-base-cased" if name not in biomed_data else "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    
    
    X, Y, L, L_llm = get_feature(name, raw_text_feature=raw_text_feature)
    input_size = None if raw_text_feature else len(X['train'][0])
    n_class = max(Y['train']) + 1

    model = MultiObjectiveModel(model_type=model_type, loss_aggr=loss_aggr, input_size=input_size, n_class=n_class,
                                learning_rate=LearnRates[lr], weight_decay=WeightDecays[wd], bert_model=bert_model)
    
    
    L_llms = [json.load(open(os.path.join(PATH, 'LLM_labels', gpt_model, 'zero-shot', name, 'train.json'))) 
              for gpt_model in gpt_models]
    # combine LLM labels
    LLM_labels = []
    for i in range(len(L['train'])):
       LLM_labels.append([L_llms[j][i][0] for j in range(len(L_llms))])
    
    model.fit(X['train'], LLM_labels, X['valid'], Y['valid'], weights=weights)
    
    Proba = {}
    for fold in X:
        Proba[fold] = model.predict_proba(X[fold]).astype('float').tolist()
        
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    with open(os.path.join(filepath, result_name), "w") as outfile:
        json.dump(Proba, outfile)
        
        
def run_harmonize_LLMs(name, gpt_models=['gpt-4', 'gpt-35-turbo', 'text-davinci-003']):
    """
    Run harmonizing LLM labels experiments with different configurations.
    """
    if len(gpt_models) == 1:
        LossAggrs = ['linear']
    else:
        LossAggrs = ['linear', 'square', '2-norm', 'max']
    for model_type in ['LR', 'MLP', 'BERT']:
        for loss_aggr in LossAggrs:
            for lr in LearnRates:
                for wd in WeightDecays:
                    harmonize_LLMs(name, gpt_models, model_type, loss_aggr, lr, wd)



def main():

    # Get GPT responses and compute transformer embeddings
    if not os.path.exists(os.path.join(PATH, 'LLM_labels')):
        os.mkdir(os.path.join(PATH, 'LLM_labels'))
    if not os.path.exists(os.path.join(PATH, 'Embeddings')):
        os.mkdir(os.path.join(PATH, 'Embeddings'))
    for name in ['cdr', 'chemprot', 'semeval', 'sms']:
        for gpt_model in ['gpt-4', 'gpt-35-turbo', 'text-davinci-003']:
            if not os.path.exists(os.path.join(PATH, 'LLM_labels', gpt_model)):
                os.mkdir(os.path.join(PATH, 'LLM_labels', gpt_model))
            if not os.path.exists(os.path.join(PATH, 'LLM_labels', gpt_model, 'zero-shot')):
                os.mkdir(os.path.join(PATH, 'LLM_labels', gpt_model, 'zero-shot'))
            X, Y, L, L_llm = get_feature(name, gpt_model=gpt_model, version='zero-shot', new_gpt_query=OpenAI_API_Config)


    # Run harmonizer training
    if not os.path.exists(os.path.join(PATH, 'Model_output')):
        os.mkdir(os.path.join(PATH, 'Model_output'))
    for name in ['cdr', 'chemprot', 'semeval', 'sms']:
        for gpt_model in ['gpt-4', 'gpt-35-turbo', 'text-davinci-003']:
            run_exp(name, gpt_model)

    
    # Dynamic prompting experiments for CDR
    for gpt_model in ['gpt-4', 'gpt-35-turbo']:
            for followup in ['reask', 'explain']:
                if not os.path.exists(os.path.join(PATH, 'LLM_labels', gpt_model, followup)):
                    os.mkdir(os.path.join(PATH, 'LLM_labels', gpt_model, followup))
                X, Y, L, L_llm = get_feature(name, gpt_model=gpt_model, version='zero-shot', new_gpt_query=False,
                                            followup=followup, OpenAI_API_Config=OpenAI_API_Config)
                
    
if __name__ == "__main__":
    main()