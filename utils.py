import json
import os
import torch
import numpy as np
from tqdm import tqdm
import retry


"""
This script contains helper functions to prepare features around the data, 
including getting GPT response on the input, and computing NLP model embeddings.
Expensive computation and querying results are stored for future use.
"""

PATH = os.path.dirname(__file__)


# GPT interaction

import openai


relation_data = ['chemprot', 'cdr', 'semeval']
biomed_data = ['chemprot', 'cdr']


prompt_settings = {'sms': 'You are an intelligent assistant to determine if a text message is spam or not spam (ham). ',
                   
                   'cdr': '''You are an intelligent assistant to extract chemical-disease relations from academic literature.
                   Your job is to determine if in the given piece of text, the drug (entity 1) induces the disease (entity 2) or not. 
                   Negative means the drug does NOT induce the disease. Positive means the drug induces the disease. Please use your judgement 
                   to the best of your knowledge. ''',
                   
                   'chemprot': '''You are an intelligent assistant to extract chemical-protein interaction from academic literature.
                   Your task is to identify the chemical-protein interactions (CHEMPROT) between entity 2: Chemical Entities Mentions (CEMs) 
                   and entity 1: Gene and Protein Related Objects (named as GPRO in the instruction below) in the given piece of text. In brief, the 
                   chemical-protein interactions include direct interactions (when a physical contact exits between a CEM and a GPRO, 
                   in most cases this GPRO being a protein or protein family and alters its function/activity) as well as indirect 
                   regulatory interactions between CEMs and GPROs (including genes, gene products (proteins, RNA), DNA/protein sequence 
                   elements and protein families, domains and complexes) that alter either the function or the quantity of the GPRO. 
                   The guidelines below provide curation rules to evaluate if the given sentence contains a description of a chemical-protein 
                   interaction; in particular, if sufficient detail/evidence is provided for comentioned CEMs and GPROs. Additionally, 
                   it describes curation rules and definitions to assign each identified chemical-protein interaction to any of the 10 classes, 
                   with detailed description listed below: 
                   0. Part of:  CEM that are structurally related to a GPRO: e.g. specific amino acid residues of a protein. 
                   1. Regulator: CEM that clearly regulates a GPRO, but for which there is no further information on whether the regulation is direct or indirect. 
                   2. Upregulator: CEM that increments a GPRO signal, without any insight on the mechanism. 
                   3. Downregulator: CEM that decreases a GPRO signal, without any insight on the mechanism. 
                   4. Agonist: CEM that binds to a receptor and alters the receptor state resulting in a biological response. 
                   5. Antagonist: CEM that reduces the action of another CEM, generally an agonist. Many antagonists act at the same receptor macromolecule as the agonist. 
                   6. Modulator: CEM that acts as allosteric modulator, compound that increases or decreases the action of an (primary or orthosteric) agonist or antagonist 
                   by combining with a distinct (allosteric or allotropic) site on the receptor macromolecule. 
                   7. Cofactor: CEM that is required for a protein's biological activity to happen. 
                   8. Substrate/Product: CEM that is both, substrate and product of enzymatic reaction. 
                   9. NOT: This class should be used to define the NEGATIVE occurrence of a chemical-protein interaction, without providing any further information on the specific negative CHEMPROT class or class.
                   Please identity the CHEMPROT interaction to the best of your knowledge. ''',
                   
                   'semeval': '''You are an intelligent assistant to help recognize semantic relations between pairs of nomimals. For example, 
                   tea and ginseng are in an ENTITY-ORIGIN relation in "The cup contained tea from dried ginseng.". You will be given a piece of text,
                   and Entity 1 and Entity 2 in the text for you to classify their semantic relation. The semantic relations are in the format of "entity1-entity2".
                   The complete semantic relation inventory is given below: 
                   0. Cause-Effect: An event or object (entity 1) leads to an effect (entity 2). Example: those cancers (entity 2) were caused by radiation exposures (entity 1)
                   1. Component-Whole: An object (entity 1) is a component of a larger whole (entity 2). Example: my apartment (entity 2) has a large kitchen (entity 1)
                   2. Content-Container: An object (entity 1) is physically stored in a delineated area of space (entity 2). Example: a bottle (entity 2) full of honey (entity 1) was weighed
                   3. Entity-Destination: An entity (entity 1) is moving towards a destination (entity 2). Example: the boy (entity 1) went to bed (entity 2)
                   4. Entity-Origin: An entity (entity 1) is coming or is derived from an origin (entity 2) (e.g., position or material). Example: letters (entity 1) from foreign countries (entity 2)
                   5. Instrument-Agency: An agent (entity 2) uses an instrument (entity 1). Example: phone (entity 1) operator (entity 2)
                   6. Member-Collection: A member (entity 1) forms a nonfunctional part of a collection (entity 2). Example: there are many trees (entity 1) in the forest (entity 2)
                   7. Message-Topic: A message (entity 1), written or spoken, is about a topic (entity 2). Example: the lecture (entity 1) was about semantics (entity 2)
                   8. Product-Producer: A producer (entity 2) causes a product (entity 1) to exist. Example: a factory (entity 2) manufactures suits (entity 1)
                   Please determine the semantic relation between entity 1 and entity 2 in the given text to the best of your knowledge. '''
                   }


# Functions to interact with the GPT models
@retry.retry(exceptions=(openai.error.APIError, openai.error.RateLimitError), tries=6, delay=1, backoff=1)
def query_gpt_model(gpt_model, setting, labels, input, examples=[], response_and_followup=None):
    """
    setting: text describing the scenario, such as 'You are an intelligent assistant to identify span messages...'
    input: the piece of text to be classified
    examples: [(text1, response1), (text2, response2), ...] Just raw text and desired response.
    """
    
    label_prompt = f'Your answer should be classified into the following categories: [{", ".join([labels[_] for _ in labels])}].'
    
    extra_prompt = """You may think step by step, articulate point by point, or make conclusion from multiple evidences, 
    but please always state the most likely label as your answer at the very begining of your response. You are encouraged to reflect on your response,
    but please keep in mind that a clear answer is always desired. Try to give a clear answer at your best guess even when you are not very sure, 
    in which case any of your conserns or explanations should go after the most likely answer to the best of your knowledge. If you are very unsure 
    about the answer and are not willing to explicitly state any label, please say 'unsure' at the very begining of your response. """
    
    example_prompt = 'Here are a few examples for you to have a better understanding of the task and learn how to correctly classify the example. '
    
    if gpt_model not in ['gpt-35-turbo', 'gpt-4']:
        # use the text completion API
        prompt = setting + label_prompt + extra_prompt
        if len(examples) > 0:
            prompt += example_prompt
            prompt += ' \n'.join(['Text: ' + ex[0] + ' \nLabel: ' + ex[1] + ' \n' for ex in examples])
        prompt += 'Text: ' + input + ' \nLabel: '
        response = openai.Completion.create(engine=gpt_model, prompt=prompt)
        output = response['choices'][0]['text']
        
    else:
        # use the chatCompletion API
        messages = [{"role": "system", "content": setting + label_prompt + extra_prompt}]
        if len(examples) > 0:
            messages[0]['content'] += example_prompt
            for ex in examples:
                messages.append({"role": "user", "content": "Please classify the following example into the most likely category: " + ex[0]})
                messages.append({"role": 'assistant', "content": ex[1]})
        messages.append({"role": "user", "content": "Please classify the following example into the most likely category: " + input})
        if response_and_followup:
            messages.append({"role": 'assistant', "content": response_and_followup[0]})
            messages.append({"role": "user", "content": response_and_followup[1] + 
                             'Are you sure about your previous answer? If not, please give a new answer. Otherwise, please restate your previous answer.'})
        
        response = openai.ChatCompletion.create(engine=gpt_model, messages=messages)
        output = response['choices'][0]['message']['content']
        
    return output


def parse_gpt_response(response, labels):
    """Parse GPT response by taking the first provided label. Abstain if model responses 'unsure' or provide no label."""
    matched_labels = {}
    for l in labels:
        ind = response.lower().find(labels[l].lower())
        if ind >= 0:
            matched_labels[int(l)] = ind
    if len(matched_labels) == 0 or 'unsure' in response.lower():
        l_gpt = -1
    else:
        l_gpt = min(matched_labels, key=matched_labels.get)
    return l_gpt





# Getting features from the WRENCH dataset
from nlp_models import get_token_ids, get_entity_masks, BERTclf, BERTre
from rule_explanations import get_explanation_from_lfs


# Main function to get all features for dataset, including GPT responses, transformer embeddings, weak labels, etc...
def get_feature(name, raw_text_feature=False, datapath=os.path.join(PATH, 'wrench'), folds=['train', 'valid', 'test'],
                gpt_model='text-davinci-003', version='zero-shot', examples=[], 
                followup=None, new_gpt_query=False, fix=False, OpenAI_API_Config=None):
    """
    name: name of the dataset
    raw_text_feature: use raw text feature or use transformer embeddings as the 'X'
    datapath: path storing the wrench datasets. Default by saving the 'wrench' folder directly under the same PATH.
    folds: the folds to return features. Defaults for all three folds.
    gpt_model: the GPT model for LLM labels, either load from storage or call API
    version: version of GPT model response
    examples: provide few-shot examples, if applicable. Default to be empty.
    followup: Option to get follow-up response using dynamic prompting. Options: 'reask', 'explain'
    new_gpt_query: configuration for making new GPT querys. Configuration format:
        {'api_key': API key, 'api_base': model endpoint, 'api_type': API type, 'api_version': API version}
    fix: option to run a pass on pre-computed GPT responses to fix any API errors
    """
    
    labels = json.load(open(os.path.join(datapath, name, 'label.json')))
    setting = prompt_settings[name]
    
    X = {fold:[] for fold in folds}
    Y = {fold:[] for fold in folds}
    L = {fold:[] for fold in folds}
    L_llm = {fold:[] for fold in folds}
    
    if new_gpt_query:
        # make completeley new queries
        fix = False
        followup = None
        
        
    if followup:
        raw_text_feature = True
        folds = ['test']
        L_llm_followup = {fold:[] for fold in folds}
        llm_followup_path = os.path.join(PATH, 'LLM_labels', gpt_model, followup, name)
        llm_followup_labeled = os.path.exists(os.path.join(llm_followup_path, 'test.json'))
        if not llm_followup_labeled:
            fix = False    # no existing label to fix
        if llm_followup_labeled and fix:
            L_llm_followup['test'] = json.load(open(os.path.join(llm_followup_path, 'test.json')))
            
    
    for fold in folds:
        data = json.load(open(os.path.join(datapath, name, f'{fold}.json')))
        
        text_based = 'text' in data[list(data)[0]]['data']
        
        # load from embedding storage
        embed_path = os.path.join(PATH, 'Embeddings', name)
        embed_computed = os.path.exists(os.path.join(embed_path, fold+'.json'))
        if embed_computed and not raw_text_feature:
            X[fold] = json.load(open(os.path.join(embed_path, fold+'.json')))
        
        # load from LLM label storage
        llm_path = os.path.join(PATH, 'LLM_labels', gpt_model, version, name)
        llm_labeled = os.path.exists(os.path.join(llm_path, fold+'.json'))
        
        if llm_labeled and not new_gpt_query:
            L_llm[fold] = json.load(open(os.path.join(llm_path, fold+'.json')))
            
        
        
        # Extract feature for each data point
        for i in tqdm(data, disable=(not new_gpt_query and not fix and not followup)):
                
            # Get X features
            if not text_based:
                # non-text-based, use precomputed feature
                X[fold].append(data[i]['data']['feature'])
            else:
                # text-based data
                if raw_text_feature or not embed_computed:
                    if name not in relation_data:
                        # classification task
                        if name != 'trec' or fold != 'test':
                            X[fold].append(data[i]['data']['text'])
                        else:
                            for c in range(len(data[i]['data']['text'])):
                                if data[i]['data']['text'][c].isupper():
                                    break
                            X[fold].append(data[i]['data']['text'][c:])
                    else:
                        # relation extraction task
                        s1 = data[i]['data']['span1']
                        s2 = data[i]['data']['span2']
                        # check valid entities
                        if (len(data[i]['data']['text'][s1[0]:s1[1]].replace(' ', '')) == 0 
                            or len(data[i]['data']['text'][s2[0]:s2[1]].replace(' ', '')) == 0):
                            continue
                        # skip long text
                        if len(data[i]['data']['text'][:s1[1]]) > 2200 or len(data[i]['data']['text'][:s2[1]]) > 2200:
                            continue
                        X[fold].append(data[i]['data'])
                    
            # Get weak-labels
            L[fold].append(data[i]['weak_labels'])
            
            # Get true labels
            Y[fold].append(data[i]['label'])
            
            # Get LLM label
            ii = len(Y[fold]) - 1
            if new_gpt_query or fix or followup:
                # new gpt query: make new GPT query to get labels. fix and followup are set to False
                # fix: fix the existing response that got API error
                # followup: get followup response based on initial response. 
                # followup + not fix: get new followup responses for all examples
                # followup + fix: fix the followup response that got API error
                # not followup + fix: fix the existing response that got API error
                
                if not new_gpt_query and not followup and L_llm[fold][ii][1]!='Error':
                    # already got gpt response
                    continue
                if not new_gpt_query and followup and fix and L_llm_followup['test'][ii][1]!='Error':
                    # already got gpt followup response
                    continue
                
                # setup OpenAI API
                if not OpenAI_API_Config:
                    raise ValueError('Please provide valid OpenAI API configuration to make GPT calls!')
                openai.api_key = OpenAI_API_Config['api_key']
                openai.api_base =  OpenAI_API_Config['api_base']
                openai.api_type = OpenAI_API_Config['api_type']
                openai.api_version = OpenAI_API_Config['api_version']
                
                # generate LLM labels querying OpenAI API
                if name not in relation_data:
                    input = data[i]['data']['text']
                else:
                    input = ' '.join([data[i]['data']['text'], 
                                      f"Entity 1: {data[i]['data']['entity1']}", 
                                      f"Entity 2: {data[i]['data']['entity2']}"])
                    
                if followup:
                    response_and_followup = [L_llm[fold][ii][1]]
                    if followup == 'explain':
                        explain = ' '.join(['It is possible that the answer could be something else. Here are some evidences to help you figure out the right answer. ', 
                                              get_explanation_from_lfs(data[i]['data'], data[i]['weak_labels'])]) 
                    else:
                        explain = ''
                    response_and_followup.append(explain)
                else:
                    response_and_followup = None
                    
                try:
                    response = query_gpt_model(gpt_model=gpt_model, 
                                               setting=setting, 
                                               labels=labels,
                                               input=input,
                                               examples=examples,
                                               response_and_followup=response_and_followup)
                    l_gpt = parse_gpt_response(response, labels)
                except:
                    l_gpt = -1
                    response = 'Error'
                    
                # record result
                if new_gpt_query:
                    L_llm[fold].append((l_gpt, response))
                else:
                    if followup and not fix:
                        L_llm_followup[fold].append([l_gpt, response] + response_and_followup)
                    elif followup and fix:
                        L_llm_followup[fold][ii] = [l_gpt, response] + response_and_followup
                    else:
                        # not followup, it must be fixing the original response
                        L_llm[fold][ii] = (l_gpt, response)
            
                
        # Save expensive computation results to storage
        
        # Store LLM labels
        if new_gpt_query or (fix and not followup):
            # updated the LLM labels, save to storage
            if not os.path.exists(llm_path):
                os.mkdir(llm_path)
            with open(os.path.join(llm_path, fold+'.json'), "w") as outfile:
                json.dump(L_llm[fold], outfile)
                
        if followup:
            if not os.path.exists(llm_followup_path):
                os.mkdir(llm_followup_path)
            with open(os.path.join(llm_followup_path, fold+'.json'), "w") as outfile:
                json.dump(L_llm_followup[fold], outfile)
                
                
        # Compute embeddings in batches and save to storage
        if text_based and not raw_text_feature and not embed_computed:
            
            X[fold] = compute_embeddings(name, X[fold])
            # Store embeddings        
            if not os.path.exists(embed_path):
                os.mkdir(embed_path)
            with open(os.path.join(embed_path, fold+'.json'), "w") as outfile:
                json.dump(X[fold], outfile)
                
    return X, Y, L, L_llm



def compute_embeddings(name, Xraw):
    """
    Compute transformer embedding from raw text input
    """
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    torch.cuda.empty_cache()
    
    bert_model = "bert-base-cased" if name not in biomed_data else "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    model = BERTclf(n_class=1, bert_model=bert_model) if name not in relation_data else BERTre(n_class=1, bert_model=bert_model)
    # n class doesn't matter for embedding computation
    model.to(device).eval()
    
    with torch.no_grad():
        X_embed = []
        batch_size=8
        n_batches = int((len(Xraw)-1) / batch_size) + 1
        for batch in tqdm(range(n_batches)):
            batch_examples = Xraw[batch*batch_size : (batch+1)*batch_size]
            if name not in relation_data:
                input_ids = get_token_ids(batch_examples, bert_model)
                input_ids = torch.as_tensor(input_ids).to(device)
                embeddings = model.get_embedding(input_ids).to(torch.device('cpu'))
            else:
                input_ids, e1_mask, e2_mask = get_entity_masks(batch_examples, bert_model)
                input_ids = torch.as_tensor(input_ids).to(device)
                e1_mask = torch.as_tensor(e1_mask).to(device)
                e2_mask = torch.as_tensor(e2_mask).to(device)
                embeddings = model.get_embedding(input_ids, e1_mask, e2_mask).to(torch.device('cpu'))
            X_embed += embeddings.tolist()
    return X_embed