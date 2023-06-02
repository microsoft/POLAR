# Supervision functions are derived from https://github.com/snorkel-team/snorkel-extraction/tree/master/tutorials/cdr

"""
This script contains functions to compile evidences from the supervision functions, which will be
provided to the GPT models for dynamic self-supervision.
The rules to signal the supervision functions was already precomputed into the weak labels.
We only provide evidences for the voted supervision functions.
"""


##### Distant supervision approaches
# We'll use the [Comparative Toxicogenomics Database](http://ctdbase.org/) (CTD) for distant supervision.
# The CTD lists chemical-condition entity pairs under three categories: therapy, marker, and unspecified.
# Therapy means the chemical treats the condition, marker means the chemical is typically present with the condition,
# and unspecified is...unspecified. We can write LFs based on these categories.

### LF_in_ctd_unspecified
def LF_in_ctd_unspecified(c):
    return 'According to the Comparative Toxicogenomics Database, the relation between the given chemical-condition pair is listed, confirming the answer. '

### LF_in_ctd_therapy
def LF_in_ctd_therapy(c):
    return f'According to the Comparative Toxicogenomics Database, the given chemical-condition pair "{c["entity1"]}-{c["entity2"]}" is listed that the chemical actually treats the condition, so the answer that {c["entity1"]} does not induce {c["entity2"]} is confirmed. '

### LF_in_ctd_marker 
def LF_in_ctd_marker(c):
    return f'According to the Comparative Toxicogenomics Database, the given chemical-condition pair "{c["entity1"]}-{c["entity2"]}" is listed that the chemical is typically present with the condition, which may confirm the answer if {c["entity1"]} induces {c["entity2"]}. '





##### Text pattern approaches
# Now we'll use some LF helpers to create LFs based on indicative text patterns.
# We came up with these rules by using the viewer to examine training candidates and noting frequent patterns.

import re

# List to parenthetical
def ltp(x):
    return '(' + '|'.join(x) + ')'

### LF_induce
def LF_induce(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_d_induced_by_c
causal_past = ['induced', 'caused', 'due']
def LF_d_induced_by_c(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_d_induced_by_c_tight
def LF_d_induced_by_c_tight(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_induce_name
def LF_induce_name(c):
    return f'The expression "{c["entity1"]}" indicates that the disease might be induced by the drug. '   

### LF_c_cause_d
causal = ['cause[sd]?', 'induce[sd]?', 'associated with']
def LF_c_cause_d(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', it is likely that {c['entity1']} induces {c['entity2']}. "


### LF_d_treat_c
treat = ['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap']
def LF_d_treat_c(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_c_treat_d
def LF_c_treat_d(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', {c['entity1']} actually treats {c['entity2']}. , so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_treat_d
def LF_treat_d(c):
    span = re.search(ltp(treat) + '.{0,50}' + c['entity2'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', {c['entity1']} actually treats {c['entity2']}, so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_c_treat_d_wide
def LF_c_treat_d_wide(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', {c['entity1']} actually treats {c['entity2']}. , so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_c_d
def LF_c_d(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', {c['entity1']} is closely mentioned with {c['entity2']}, so they should be closely related. "

### LF_c_induced_d
def LF_c_induced_d(c):
    start = min(c['span1'][0], c['span2'][0])
    end = max(c['span1'][1], c['span2'][1])
    return f"Based on the expression '{c['text'][start:end]}', {c['entity1']} is closely mentioned with {c['entity2']}. , so it is likely that {c['entity1']} induces {c['entity2']}. "


### LF_improve_before_disease
def LF_improve_before_disease(c):
    span = re.search('improv.*' + re.escape(c['entity2']), c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', the disease {c['entity2']} is actually improved, so it is not likely that {c['entity1']} induces {c['entity2']}. "


### LF_in_patient_with
pat_terms = ['in a patient with ', 'in patients with']
def LF_in_patient_with(c):
    span = re.search(ltp(pat_terms) + '.{0,5}' + c['entity2'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', {c['entity2']} is the initial condition of the patient(s), so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_uncertain
uncertain = ['combin', 'possible', 'unlikely']
def LF_uncertain(c):
    span = re.search(ltp(uncertain), c['text'], re.IGNORECASE).span()
    ends = []
    if c['span1'][1] > span[1]:
        ends.append(c['span1'][1])
    if c['span2'][1] > span[1]:
        ends.append(c['span2'][1])
    evidence = c['text'][span[0] : min(ends)]
    return f"Based on the expression '{evidence}', it is uncertain that {c['entity1']} induces {c['entity2']}. "


### LF_induced_other
def LF_induced_other(c):
    span = re.search('-induced' + '.{0,5}' + c['entity2'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', {c['entity2']} is induced by other factors, so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_far_c_d
def LF_far_c_d(c):
    return f"{c['entity1']} and {c['entity2']} are not closely mentioned in the text, so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_far_d_c
def LF_far_d_c(c):
    return f"{c['entity1']} and {c['entity2']} are not closely mentioned in the text, so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_risk_d
def LF_risk_d(c):
    span = re.search('risk of' + '.{0,5}' + c['entity2'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_develop_d_following_c
def LF_develop_d_following_c(c):
    span = re.search('develop.{0,25}' + c['entity2'] + '.{0,25}following.{0,25}' + c['entity1'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_d_following_c
procedure, following = ['inject', 'administrat'], ['following']
def LF_d_following_c(c):
    span = re.search(c['entity2'] + '.{0,50}' + ltp(following) + '.{0,20}' + c['entity1'] + '.{0,50}' + ltp(procedure), c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is likely that {c['entity1']} induces {c['entity2']}. "

### LF_measure
def LF_measure(c):
    span = re.search('measur.{0,75}' + re.escape(c['entity1']), c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is not likely that {c['entity1']} induces {c['entity2']}. "


### LF_level
def LF_level(c):
    span = re.search(c['entity1'] + '.{0,25} level', c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is not likely that {c['entity1']} induces {c['entity2']}. "


### LF_neg_d
def LF_neg_d(c):
    if re.search('(none|not|no) .{0,25}' + c['entity1'], c['text'], re.IGNORECASE):
        span = re.search('(none|not|no) .{0,25}' + c['entity1'], c['text'], re.IGNORECASE).span()
    elif re.search('(none|not|no) .{0,25}' + c['entity2'], c['text'], re.IGNORECASE):
        span = re.search('(none|not|no) .{0,25}' + c['entity2'], c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"Based on the expression '{evidence}', it is not likely that {c['entity1']} induces {c['entity2']}. "


### LF_weak_assertions
WEAK_PHRASES = ['none', 'although', 'was carried out', 'was conducted',
                'seems', 'suggests', 'risk', 'implicated',
               'the aim', 'to (investigate|assess|study)']

WEAK_RGX = r'|'.join(WEAK_PHRASES)
def LF_weak_assertions(c):
    span = re.search(WEAK_RGX, c['text'], re.IGNORECASE).span()
    evidence = c['text'][span[0] : span[1]]
    return f"According to phrases like '{evidence}', there is no strong signal that {c['entity1']} induces {c['entity2']}. "





##### Composite LFs

# The following LFs take some of the strongest distant supervision and text pattern LFs,
# and combine them to form more specific LFs. These LFs introduce some obvious 
# dependencies within the LF set, which we will model later.

### LF_ctd_marker_c_d
def LF_ctd_marker_c_d(c):
    return LF_in_ctd_marker(c)

### LF_ctd_marker_induce
def LF_ctd_marker_induce(c):
    return LF_in_ctd_marker(c)

### LF_ctd_therapy_treat
def LF_ctd_therapy_treat(c):
    return LF_in_ctd_therapy(c)

### LF_ctd_unspecified_treat
def LF_ctd_unspecified_treat(c):
    return LF_in_ctd_unspecified(c)

### LF_ctd_unspecified_induce
def LF_ctd_unspecified_induce(c):
    return LF_in_ctd_unspecified(c)






##### Rules based on context hierarchy
# These last two rules will make use of the context hierarchy.
# The first checks if there is a chemical mention much closer to the candidate's disease mention
# than the candidate's chemical mention. The second does the analog for diseases.

### LF_closer_chem
def LF_closer_chem(c):
    return f"According to the text, another chemical is mentioned closer to {c['entity2']} than {c['entity1']}, so it is not likely that {c['entity1']} induces {c['entity2']}. "

### LF_closer_dis
def LF_closer_dis(c):
    return f"According to the text, another disease is mentioned closer to {c['entity1']} than {c['entity2']}, so it is not likely that {c['entity1']} induces {c['entity2']}. "



LFs = [
    LF_c_cause_d,
    LF_c_d,
    LF_c_induced_d,
    LF_c_treat_d,
    LF_c_treat_d_wide,
    LF_closer_chem,
    LF_closer_dis,
    LF_ctd_marker_c_d,
    LF_ctd_marker_induce,
    LF_ctd_therapy_treat,
    LF_ctd_unspecified_treat,
    LF_ctd_unspecified_induce,
    LF_d_following_c,
    LF_d_induced_by_c,
    LF_d_induced_by_c_tight,
    LF_d_treat_c,
    LF_develop_d_following_c,
    LF_far_c_d,
    LF_far_d_c,
    LF_improve_before_disease,
    LF_in_ctd_therapy,
    LF_in_ctd_marker,
    LF_in_patient_with,
    LF_induce,
    LF_induce_name,
    LF_induced_other,
    LF_level,
    LF_measure,
    LF_neg_d,
    LF_risk_d,
    LF_treat_d,
    LF_uncertain,
    LF_weak_assertions,
]


def _append_explanation(j, x, l, explain):
    """
    Helper function to get explanation for the supervision function j on example x. l is the label from SF j.
    """
    if l < 0:
        return explain
    evidence = LFs[j](x)
    if evidence not in explain:
        explain += evidence
    return explain

"""
Main function to call to compile the evidences.
The rules to signal the supervision functions was already precomputed into the weak labels.
We only provide evidences for the voted supervision functions.
"""
def get_explanation_from_lfs(x, L):
    explain = ''
    for j in range(len(L)):
        if j in [1, 7]:
            for jj in [1, 7]:
                explain = _append_explanation(jj, x, L[jj], explain)
        elif j in [2, 14, 8]:
            for jj in [2, 14, 8, 11]:
                explain = _append_explanation(jj, x, L[jj], explain)
        elif j in [4, 9, 10]:
            for jj in [4, 9, 10]:
                explain = _append_explanation(jj, x, L[jj], explain)
        else:
            explain = _append_explanation(j, x, L[j], explain)
    return explain
                