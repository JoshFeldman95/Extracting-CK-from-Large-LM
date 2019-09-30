import sys
import csv

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from sentences import DirectTemplate, PredefinedTemplate, EnumeratedTemplate
from knowledge_miner import KnowledgeMiner
from pytorch_pretrained_bert import BertForMaskedLM, GPT2LMHeadModel

bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'

template_repo = './templates/'
single_templates= 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'

data_repo = './data/'
test_data = 'test.txt'

def get_tuned_f1(results_df):
    """ Tune PMI weight with separability and return optimal lambda and f1 score values """
    df = results_df
    scores = []
    f1s = []
    lambdas = np.arange(.5, 5, .05)

    def add_weighted(df, lam):
        """ Calculates different weighted PMI values after already having mut_inf scores """
        df['mut_inf_weighted'] = df.mut_inf + (lam - 1) * (df.head_conditional + df.tail_conditional) / 2.

    for lam in lambdas:
        ss = StandardScaler()
        add_weighted(df, lam=lam)
        model = GaussianMixture(2, n_init=1)
        dat = ss.fit_transform(df[['mut_inf_weighted']])
        pred = model.fit_predict(dat)
        score = model.aic(dat)
        f1 = f1_score((model.means_.argmax() == df.label), pred)
        scores.append(score)
        f1s.append(f1)

    scores = np.array(scores)
    f1s = np.array(f1s)
    lam = lambdas[scores.argmax()]

    optimal_lambda = lambdas[scores.argmax()]
    optimal_f1 = f1s[scores.argmax()]
    return optimal_f1, optimal_lambda
    

def run_experiment(template_type, knowledge_miners):
    print(f'make predictions using {template_type} templates...')
    ck_miner = knowledge_miners[template_type]

    df = ck_miner.make_predictions()

    print(f'saving results as {data_repo}ckbc_predictions_{template_type}.csv')
    df.to_csv(data_repo + f'ckbc_predictions_{template_type}.csv')

    f1, lam = get_tuned_f1(df)
    print(f'CKBC {template_type} Lambda: {lam}, F1 Score: {f1}')


def mine(hardware):
    print('loading BERT...')
    bert = BertForMaskedLM.from_pretrained(bert_model)
    print('loading GPT2...')
    gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)

    knowledge_miners = {
        'concat': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            DirectTemplate,
            bert
        ),
        'template': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            PredefinedTemplate,
            bert,
            grammar = False,
            template_loc = template_repo + single_templates
        ),
        'template_grammar': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            PredefinedTemplate,
            bert,
            grammar = True,
            template_loc = template_repo + single_templates
        ),
        'coherency': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            EnumeratedTemplate,
            bert,
            language_model = gpt,
            template_loc = template_repo + multiple_templates
        )
    }

    for template_type in knowledge_miners.keys():
        run_experiment(template_type, knowledge_miners)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ckbc_experiments.py -<cuda or cpu>')
    else:
        hardware = sys.argv[1].replace('-', '', 1)
        mine(hardware)
