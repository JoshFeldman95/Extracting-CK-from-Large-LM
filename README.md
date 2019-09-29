# Commonsense Knowledge Mining from Pretrained Models
This repo contains the experiments from the paper "Commonsense Knowledge Mining from Pretrained Models" by Joshua Feldman, Joe Davison, and Sasha Rush

# Usage
## Commonsense Knowledge Base Completion Experiments
**to do**
## Wikipedia Experiments

To reproduce the wikipedia experiments, run `python wikipedia_experiments.py -cpu` or `python wikipedia_experiments.py -cuda`.

This script will put 4 datasets in the `data` repo: `wiki_concat.csv`,`wiki_template.csv`,`wiki_template_grammar.csv`, and `wiki_coherency.csv`. These files correspond to the concatenation, template, template + grammar, and coherency ranking experiments in the paper. The data fields are defined as follows:
- **nll**: The negative log likelihood (p(T|H,r)/len(T)).
- **tail_conditional**: The conditional probability of the tail given the head (p(T|H,r)).
- **head_conditional**: The conditional probability of the head given the tail (p(H|T,r)).
- **tail_marginal**: The marginal probability of the tail (p(T|r)).
- **head_marginal**: The marginal probability of the head (p(H|r)).
- **mut_inf**: The mutual information ((p(T|H,r)-p(T|r) + p(H|T,r)-p(H|r))/2.
- **label**: If triple has a label, 1 = valid and 0 = invalid. Wikipedia data do not have labels and you can ignore this field.
- **sent**: The generated sentence fed to BERT.
