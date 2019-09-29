import sys
from experiments import DirectTemplate, PredefinedTemplate, EnumeratedTemplate, KnowledgeMiner
from pytorch_pretrained_bert import BertForMaskedLM, GPT2LMHeadModel

bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'

template_repo = './templates/'
single_templates= 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'

data_repo = './data/'
wikipedia_candidates = 'NovelTuples_debug.csv'

def run_experiment(template_type, knowledge_miners):
    print(f'make predictions using {template_type} templates...')
    ck_miner = knowledge_miners[template_type]

    df = ck_miner.make_predictions()

    print(f'saving results as {data_repo}wikipedia_predictions_{template_type}.csv')
    df.to_csv(data_repo + f'wikipedia_predictions_{template_type}.csv')


def mine_from_wikipedia(hardware):
    print('loading BERT...')
    bert = BertForMaskedLM.from_pretrained(bert_model)
    print('loading GPT2...')
    gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)

    knowledge_miners = {
        'concat': KnowledgeMiner(
            data_repo + wikipedia_candidates,
            hardware,
            DirectTemplate,
            bert
        ),
        'template': KnowledgeMiner(
            data_repo + wikipedia_candidates,
            hardware,
            PredefinedTemplate,
            bert,
            grammar = False,
            template_loc = template_repo + single_templates
        ),
        'template_grammar': KnowledgeMiner(
            data_repo + wikipedia_candidates,
            hardware,
            PredefinedTemplate,
            bert,
            grammar = True,
            template_loc = template_repo + single_templates
        ),
        'coherency': KnowledgeMiner(
            data_repo + wikipedia_candidates,
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
        print('Usage: python wikipedia_experiments.py -<cuda or cpu>')
    else:
        hardware = sys.argv[1].replace('-', '', 1)
        mine_from_wikipedia(hardware)
