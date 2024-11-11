import json
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


def chunk_continuation(model, tokenizer, context, prefix, candidate_set, suffix, sum=False, verbose=False):
    full_prefix = context + ' ' + prefix
    input_ids = tokenizer(full_prefix, return_tensors='pt').input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        past_key_values = outputs.past_key_values

    log_probs = []
    for candidate in candidate_set:
        candidate_ids = tokenizer(candidate, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(candidate_ids, past_key_values=past_key_values)
            logits = outputs.logits
            token_log_probs = logits[0, -1, :].log_softmax(dim=-1)
            if sum:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).sum().item())
            else:
                log_probs.append(token_log_probs.index_select(0, candidate_ids[0, :]).mean().item())

    if verbose:
        for candidate, log_prob in zip(candidate_set, log_probs):
            print(f"{candidate}: {log_prob}")    

    probs = torch.softmax(torch.tensor(log_probs), dim=0)
    sampled_cand = torch.multinomial(probs, 1).item()
    if verbose:
        print(f"Sampled candidate: {candidate_set[sampled_cand]}")

    sample = {
        'sampled_text': f'{prefix}{candidate_set[sampled_cand]}{suffix}',
        'full_text': f'{full_prefix}{candidate_set[sampled_cand]}{suffix}',
        'sampled_index': sampled_cand,
        'candidate_logprobs': log_probs
    }

    return sample


def sample_sequence(model, tokenizer, sequence_sample_space):
    prefix = ''
    sampled_indices = []
    sampled_logprobs = []
    sampled_text_list = []
    variable_names = []
    for setup_dict in sequence_sample_space:
        variable_names.append(setup_dict['variable_name'])
        context = ''
        if setup_dict['parent_indices'] is not None:
            for parent_index in range(len(sequence_sample_space)):
                if parent_index in setup_dict['parent_indices']:
                    context = context + ' ' + sampled_text_list[parent_index]
        if setup_dict['intervention_choice'] is None:
            chunk_sample = chunk_continuation(
                model=model, 
                tokenizer=tokenizer,
                context=context, 
                prefix=setup_dict['prefix'], 
                candidate_set=setup_dict['candidate_set'],
                suffix=setup_dict['suffix'],
                sum=False,
                verbose=False
            )
            sampled_text = chunk_sample['sampled_text']
            sampled_index = chunk_sample['sampled_index']
            candidate_logprobs = chunk_sample['candidate_logprobs']
        elif setup_dict['intervention_choice'] == 'uniform':
            candidate_set = setup_dict['candidate_set']
            sampled_index = np.random.randint(low=0, high=len(candidate_set))
            prefix = setup_dict['prefix']
            suffix = setup_dict['suffix']
            sampled_text = f'{prefix}{candidate_set[sampled_index]}{suffix}'
            candidate_logprobs = (np.ones_like(candidate_set, dtype=float) * float('-inf')).tolist()
            candidate_logprobs[sampled_index] = 0
        elif isinstance(setup_dict['intervention_choice'], int):
            sampled_index = setup_dict['intervention_choice']
            candidate_set = setup_dict['candidate_set']
            prefix = setup_dict['prefix']
            suffix = setup_dict['suffix']
            sampled_text = f'{prefix}{candidate_set[sampled_index]}{suffix}'
            candidate_logprobs = (np.ones_like(candidate_set, dtype=float) * float('-inf')).tolist()
            candidate_logprobs[sampled_index] = 0
        else:
            raise ValueError(f"Unsupported type for 'intervention_choice': {setup_dict['intervention_choice']}")
        
        sampled_text_list.append(sampled_text)
        sampled_indices.append(sampled_index)
        sampled_logprobs.append(candidate_logprobs)

    sample = {
        'sampled_text': ' '.join(sampled_text_list),
        'sampled_indices': sampled_indices,
        'sampled_logprobs': sampled_logprobs,
        'variable_names': variable_names,
    }
            
    return sample


def sample_interventional_sequence(model, tokenizer, sequence_sample_space, index_of_intervention, intervention_choice):
    modified_sample_space = deepcopy(sequence_sample_space)
    
    if modified_sample_space[index_of_intervention]['exogenous']:
        raise ValueError('Cannot use an exogenous variable as the intervention.')
    
    for i in range(len(sequence_sample_space)):
        modified_sample_space[i]['intervention_choice'] = None

    modified_sample_space[index_of_intervention]['intervention_choice'] = intervention_choice
    
    return sample_sequence(model, tokenizer, modified_sample_space)


def sample_conditional_interventional_sequence(model, tokenizer, sequence_sample_space, condition_dict, index_of_intervention, intervention_choice):
    modified_sample_space = deepcopy(sequence_sample_space)
    
    if modified_sample_space[index_of_intervention]['exogenous']:
        raise ValueError('Cannot use an exogenous variable as the intervention.')
    
    for i in range(len(modified_sample_space)):
        if i in condition_dict.keys():
            if modified_sample_space[i]['exogenous']:
                raise ValueError('Cannot condition on an exogenous variable unless sampling counterfactuals.')
            modified_sample_space[i]['intervention_choice'] = condition_dict[i]
        else:
            modified_sample_space[i]['intervention_choice'] = None

    modified_sample_space[index_of_intervention]['intervention_choice'] = intervention_choice
    
    return sample_sequence(model, tokenizer, modified_sample_space)


def sample_counterfactual_sequence(model, tokenizer, sequence_sample_space, sampled_indices, evidence_indices, index_of_intervention, intervention_choice):
    modified_sample_space = deepcopy(sequence_sample_space)
    
    for i in range(len(sequence_sample_space)):
        if modified_sample_space[i]['exogenous'] or i in evidence_indices:
            modified_sample_space[i]['intervention_choice'] = sampled_indices[i]
        else:
            modified_sample_space[i]['intervention_choice'] = None

    if modified_sample_space[index_of_intervention]['exogenous']:
        raise ValueError('Cannot use an exogenous variable as the intervention.')
    modified_sample_space[index_of_intervention]['intervention_choice'] = intervention_choice
    
    return sample_sequence(model, tokenizer, modified_sample_space)


def sample_sequences(model, tokenizer, sequence_sample_space, num_samples, verbose=True):
    samples = []
    for _ in tqdm(range(num_samples), disable=not verbose):
        sample = sample_sequence(model, tokenizer, sequence_sample_space)
        samples.append(sample)
    return samples


def sample_interventional_sequences(model, tokenizer, sequence_sample_space, num_samples, index_of_intervention, intervention_choice, verbose=True):
    samples = []
    for _ in tqdm(range(num_samples), disable=not verbose):
        sample = sample_interventional_sequence(model, tokenizer, sequence_sample_space, index_of_intervention, intervention_choice)
        samples.append(sample)
    return samples


def sample_conditional_interventional_sequences(model, tokenizer, sequence_sample_space, num_samples, condition_dict, index_of_intervention, intervention_choice, verbose=True):
    samples = []
    for _ in tqdm(range(num_samples), disable=not verbose):
        sample = sample_conditional_interventional_sequence(model, tokenizer, sequence_sample_space, condition_dict, index_of_intervention, intervention_choice)
        samples.append(sample)
    return samples


def sample_observed_conditional_interventional_sequences(model, tokenizer, sequence_sample_space, samples, indices_to_condition, index_of_intervention, intervention_choice, verbose=True):
    conditional_samples = []
    for sample in tqdm(samples, disable=not verbose):
        condition_dict = {index: sample['sampled_indices'][index] for index in indices_to_condition}
        sample = sample_conditional_interventional_sequence(model, tokenizer, sequence_sample_space, condition_dict, index_of_intervention, intervention_choice)
        conditional_samples.append(sample)
    return conditional_samples


def sample_counterfactual_sequences(model, tokenizer, sequence_sample_space, samples, evidence_indices, index_of_intervention, intervention_choice, verbose=True):
    cf_samples = []
    for sample in tqdm(samples, disable=not verbose):
        cf_sample = sample_counterfactual_sequence(model, tokenizer, sequence_sample_space, sample['sampled_indices'], evidence_indices, index_of_intervention, intervention_choice)
        cf_samples.append(cf_sample)
    return cf_samples


def plot_dag_from_sample_space(sequence_sample_space, title=None):
    graph = nx.DiGraph()
    for setup_dict in sequence_sample_space:
        node_name = setup_dict['variable_name']
        graph.add_node(node_name) 
        if setup_dict['parent_indices'] is not None:
            for parent_index in setup_dict['parent_indices']:
                parent_name = sequence_sample_space[parent_index]['variable_name']
                graph.add_edge(parent_name, node_name)
    
    options = {
        'node_color': 'white',
        'alpha': 1,
        'node_size': 5000,
        'width': 3,
        'arrowstyle': '-|>',
        'arrowsize': 12,
    }
    nx.draw_networkx(graph, pos=nx.circular_layout(graph), arrows=True, **options)
    if title:
        plt.title(title)
    plt.show()
    

def format_sequences_as_dataframe(sequences, name_of_intervention, name_of_outcome, save_logprobs=False):
    df_rows = []
    for sequence in sequences:
        df_row = {
            'sampled_text': sequence['sampled_text']
        }
        df_row.update(dict(zip(sequence['variable_names'], sequence['sampled_indices'])))
        
        for i, name in enumerate(sequence['variable_names']):
            if save_logprobs or name in [name_of_intervention, name_of_outcome]:
                if name == name_of_outcome:
                    possible_outcome_values = range(len(sequence['sampled_logprobs'][i]))
                if name == name_of_intervention:
                    possible_treatment_values = range(len(sequence['sampled_logprobs'][i]))
                logprob_names = [f'logP({name}={value})' for value in range(len(sequence['sampled_logprobs'][i]))]
                df_row.update(dict(zip(logprob_names, sequence['sampled_logprobs'][i])))
        
        df_rows.append(df_row)
    data_df = pd.DataFrame(df_rows)
    
    logprob_treatment_names = [f'logP({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
    prob_treatment_names = [f'P({name_of_intervention}={treatment_value})' for treatment_value in possible_treatment_values]
    data_df[prob_treatment_names] = torch.softmax(torch.tensor(data_df[logprob_treatment_names].values), dim=1)
    
    logprob_outcome_names = [f'logP({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
    prob_outcome_names = [f'P({name_of_outcome}={outcome_value})' for outcome_value in possible_outcome_values]
    data_df[prob_outcome_names] = torch.softmax(torch.tensor(data_df[logprob_outcome_names].values), dim=1)
    
    return data_df


def sample_setup_dict(config_path, config_random_state):
    np.random.seed(config_random_state)
    torch.manual_seed(config_random_state)
    
    setup_dict = {}
    with open(config_path) as json_file:
        json_data = json.load(json_file)
        sequence_sample_space = json_data['setup_sequence_sample_space']
        chosen_sample_space = []
        for covariate_sample_space in sequence_sample_space:
            sequence_dict = {}
            possible_prefix_suffix_pairs = covariate_sample_space['possible_prefix_suffix_pairs']
            chosen_index = np.random.randint(low=0, high=len(possible_prefix_suffix_pairs), size=1)[0]
            sequence_dict['prefix'] = possible_prefix_suffix_pairs[chosen_index][0]
            sequence_dict['suffix'] = possible_prefix_suffix_pairs[chosen_index][1]
            sequence_dict['candidate_set'] = covariate_sample_space['candidate_set']
            sequence_dict['intervention_choice'] = covariate_sample_space['intervention_choice']
            sequence_dict['parent_indices'] = covariate_sample_space['parent_indices']
            sequence_dict['exogenous'] = covariate_sample_space['exogenous']
            sequence_dict['variable_name'] = covariate_sample_space['variable_name']
            chosen_sample_space.append(sequence_dict)
            
        setup_dict['sequence_sample_space'] = chosen_sample_space
        setup_dict['index_of_intervention'] = json_data['index_of_intervention']
        setup_dict['intervention_choices'] = json_data['intervention_choices']
        setup_dict['index_of_outcome'] = json_data['index_of_outcome']
        setup_dict['possible_outcome_choices'] = json_data['possible_outcome_choices']
        
    return setup_dict


def generate_data_from_config(model, tokenizer, config_path, config_random_state, sample_random_state, num_samples, verbose=True):
    setup_dict = sample_setup_dict(config_path=config_path, config_random_state=config_random_state)
    sequence_sample_space = setup_dict['sequence_sample_space']
    index_of_intervention = setup_dict['index_of_intervention']
    intervention_choices = setup_dict['intervention_choices']
    index_of_outcome = setup_dict['index_of_outcome']
    possible_outcome_choices = setup_dict['possible_outcome_choices']

    if verbose:
        print('Chosen sample space:')
        print(sequence_sample_space)

    np.random.seed(sample_random_state)
    torch.manual_seed(sample_random_state)
    
    samples = sample_sequences(model=model, tokenizer=tokenizer, sequence_sample_space=sequence_sample_space, num_samples=num_samples, verbose=verbose)

    observed_feature_names = [f'X{i}' for i in range(len(sequence_sample_space))]
    observed_data = pd.DataFrame([sample['sampled_indices'] for sample in samples], columns=observed_feature_names)
    
    for intervention_choice in intervention_choices:
        propensity_name = f'logP(X{index_of_intervention}={intervention_choice})'
        observed_data[propensity_name] = [sample['sampled_logprobs'][index_of_intervention][intervention_choice] for sample in samples]
    
    for outcome_choice in possible_outcome_choices:
        outcome_name = f'logP(X{index_of_outcome}={outcome_choice})'
        observed_data[outcome_name] = [sample['sampled_logprobs'][index_of_outcome][outcome_choice] for sample in samples]
    
    data_frames_to_join = [observed_data]
    for intervention_choice in intervention_choices:
        cf_samples = sample_counterfactual_sequences(
            model=model, 
            tokenizer=tokenizer, 
            sequence_sample_space=sequence_sample_space,
            samples=samples, 
            evidence_indices=list(range(index_of_intervention)),
            index_of_intervention=index_of_intervention, 
            intervention_choice=intervention_choice,
            verbose=verbose
        )

        feature_names = [
            f'X{i}|do(X{index_of_intervention}={intervention_choice})' 
            for i in range(len(sequence_sample_space))
        ]
        counterfactual_df = pd.DataFrame([sample['sampled_indices'] for sample in cf_samples], columns=feature_names)

        for outcome_choice in possible_outcome_choices:
            outcome_name = f'logP(X{index_of_outcome}={outcome_choice})|do(X{index_of_intervention}={intervention_choice})'
            counterfactual_df[outcome_name] = [sample['sampled_logprobs'][index_of_outcome][outcome_choice] for sample in cf_samples]
        
        data_frames_to_join.append(counterfactual_df)
    
    data_df = pd.concat(data_frames_to_join, axis=1)
    
    result_dict = {
        'data_df': data_df,
        'sequence_sample_space': sequence_sample_space,
        'samples': samples
    }
    
    return result_dict
