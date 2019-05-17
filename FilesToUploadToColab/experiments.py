from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, \
                                    GPT2LMHeadModel, GPT2Tokenizer
import torch
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from nltk import pos_tag
import nltk
import pandas as pd
from pathlib import Path
from copy import deepcopy
import spacy
# pip install pattern should work
from pattern.en import conjugate, PARTICIPLE, referenced, INDEFINITE, pluralize

class CommonsenseTuples(Dataset):
    def __init__(self, tuple_dir, mask_head, mask_tail, device, language_model = None):
        """
        Args:
            tuple_dir (string): Path to the csv file with commonsense tuples
        """
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.mask_head = mask_head
        self.mask_tail = mask_tail

        self.sep_token = '[SEP]'
        self.start_token = '[CLS]'
        self.mask_token = '[MASK]'
        self.pad_token = '[PAD]'

        self.max_len = 20
        self.stop_tokens = ['the', 'a', 'an']
        self.device = device

        self.model = language_model
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

        # Load tuples
        with open(tuple_dir) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            self.tuples = [row for row in reader]

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        # get tuple
        r, t1, t2, label = self.tuples[idx][:4]

        # apply template
        #try:
        sent, t1, t2 = self.apply_template(r, t1, t2)
        #except (json.JSONDecodeError) as e:
        #    return (-1,-1,-1,-1)
        # apply start and end tokens
        sent = f"{self.start_token} {sent}. {self.sep_token}"

        # tokenize sentences and t1 and t2
        tokenized_sent = self.tokenizer.tokenize(sent)
        tokenized_t1 = self.tokenizer.tokenize(t1)
        tokenized_t2 = self.tokenizer.tokenize(t2)

        # mask sentence
        tokenized_masked = self.mask_sentence(tokenized_sent, tokenized_t1, tokenized_t2)

        # get segment ids
        segments_ids = self.get_segment_ids(tokenized_sent)

        # convert tokens to ids
        indexed_sent = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        indexed_masked = self.tokenizer.convert_tokens_to_ids(tokenized_masked)

        return (
            torch.tensor(indexed_sent, device=self.device),
            torch.tensor(indexed_masked, device=self.device),
            torch.tensor(segments_ids, device=self.device),
            int(label)
        )

    def mask(self, tokenized_sent, tokenized_to_mask):
        tokenized_masked = tokenized_sent.copy()
        for idx_sent in range(len(tokenized_masked)-len(tokenized_to_mask)):
            match = []
            for idx_mask in range(len(tokenized_to_mask)):
                match.append(tokenized_masked[idx_sent+idx_mask] == tokenized_to_mask[idx_mask])
            if all(match):
                for idx_mask in range(len(tokenized_to_mask)):
                    if tokenized_masked[idx_sent+idx_mask] not in self.stop_tokens:
                        tokenized_masked[idx_sent+idx_mask] = self.mask_token
        return tokenized_masked

    def mask_sentence(self, tokenized_sent, tokenized_t1, tokenized_t2):
        if self.mask_head:
            tokenized_sent = self.mask(tokenized_sent, tokenized_t1)
        if self.mask_tail:
            tokenized_sent = self.mask(tokenized_sent, tokenized_t2)
        return tokenized_sent

    def get_segment_ids(self, tokenized_sent):
        segments_ids = []
        segment = 0
        for word in tokenized_sent:
            segments_ids.append(segment)
            if word == self.sep_token:
                segment += 1
        return segments_ids

    def id_to_text(self, sent):
        if type(sent) == torch.Tensor:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx].item()] for idx in range(len(sent))]
        else:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx]] for idx in range(len(sent))]
        return " ".join(tokens)

    def apply_template(self, relation, head, tail):
        """ To be overriden, returning the sentence, head, and tail """
        return


class DirectTemplate(CommonsenseTuples):

    def __init__(self, *args):
        super().__init__(*args)
        self.regex = '[A-Z][^A-Z]*'

    def apply_template(self, relation, head, tail):
        template = " ".join(re.findall(self.regex, relation))
        return ' '.join([head, template, tail]), head, tail


class PredefinedTemplate(CommonsenseTuples):

    def __init__(self, *args, template_loc='relation_map.json', grammar = False):
        super().__init__(*args)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        template_loc  = Path(__file__).parent / template_loc
        self.grammar = grammar
        with open(template_loc, 'r') as f:
            self.templates = json.load(f)

    def clean_text(self, words):
        new_words = words.split(' ')
        doc = self.nlp(words)
        first_word_POS = doc[0].pos_
        if first_word_POS == 'VERB':
            new_words[0] = conjugate(new_words[0], tense=PARTICIPLE)
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            if new_words[0] != 'a' or new_words[0] != 'an':
                new_words[0] = referenced(new_words[0])
        elif first_word_POS == 'NUM' and len(new_words) > 1:
            new_words[1] = pluralize(new_words[1])
        return ' '.join(new_words)

    def apply_template(self, relation, head, tail):
        if self.grammar:
            head = self.clean_text(head)
            tail = self.clean_text(tail)
        sent = self.templates[relation].format(head, tail)
        return sent, head, tail

class SurfaceTexts(CommonsenseTuples):
    def __init__(self, *args):
        super().__init__(*args)

    def apply_template(self, sent, head, tail):
        head_or_tail = re.findall('\[\[.+?\]\]', sent)

        head_new = [text for text in head_or_tail if head.split(' ')[0] in text.lower()][0]
        tail_new = [text for text in head_or_tail if tail.split(' ')[0] in text.lower()][0]

        head_clean, tail_clean, sent_clean = [re.sub('[\[\]\*\.]','', text) for text in (head_new, tail_new, sent)]

        return sent_clean, head_clean, tail_clean

    def __getitem__(self, idx):
        # get tuple
        r, t1, t2, label, sent = self.tuples[idx][:5]

        # apply template
        try:
            sent, t1, t2 = self.apply_template(sent, t1, t2)
        except (TypeError, json.JSONDecodeError) as e:
            return (-1,-1,-1,-1)
        # apply start and end tokens
        sent = f"{self.start_token} {sent}. {self.sep_token}"

        # tokenize sentences and t1 and t2
        tokenized_sent = self.tokenizer.tokenize(sent)
        tokenized_t1 = self.tokenizer.tokenize(t1)
        tokenized_t2 = self.tokenizer.tokenize(t2)

        # mask sentence
        tokenized_masked = self.mask_sentence(tokenized_sent, tokenized_t1, tokenized_t2)

        # get segment ids
        segments_ids = self.get_segment_ids(tokenized_sent)

        # convert tokens to ids
        indexed_sent = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        indexed_masked = self.tokenizer.convert_tokens_to_ids(tokenized_masked)

        return torch.tensor(indexed_sent), torch.tensor(indexed_masked),\
                    torch.tensor(segments_ids), int(label)

class EnumeratedTemplate(CommonsenseTuples):
    def __init__(self, *args, language_model = None, template_loc='./relation_map_multiple.json'):
        super().__init__(*args, language_model = language_model)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.enc = GPT2Tokenizer.from_pretrained('gpt2')

        with open(template_loc, 'r') as f:
            self.templates = json.load(f)

    def apply_template(self, relation, head, tail):
        candidate_sents = self.get_candidates(relation, head, tail)
        sent, head, tail = self.get_best_candidate(candidate_sents)
        return sent, head, tail

    def get_candidates(self, relation, head, tail):
        heads = self.formats(head)
        tails = self.formats(tail)
        templates = self.templates[relation]
        candidate_sents = []

        for h in heads:
            for t in tails:
                for temp in templates:
                    candidate_sents.append((temp.format(h, t), h, t))
        return candidate_sents

    def formats(self, phrase):
        doc = self.nlp(phrase)
        first_word_POS = doc[0].pos_

        tokens = phrase.split(' ')
        new_tokens = tokens.copy()

        new_phrases = []
        # original
        new_phrases.append(' '.join(new_tokens))

        # with indefinite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = referenced(tokens[0])
            new_phrases.append(' '.join(new_tokens))
        # with definite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = "the "+tokens[0]
            new_phrases.append(' '.join(new_tokens))
        # as gerund
        if first_word_POS == 'VERB':
            new_tokens[0] = conjugate(tokens[0], tense=PARTICIPLE)
            new_phrases.append(' '.join(new_tokens))
            if len(tokens) > 1:
                if tokens[1] == 'to' and len(tokens) > 2:
                    new_tokens[2] = referenced(tokens[2])
                else:
                    new_tokens[1] = referenced(tokens[1])
            new_phrases.append(' '.join(new_tokens))
            new_tokens[0] = tokens[0]
            new_phrases.append(' '.join(new_tokens))


        # account for numbers
        if first_word_POS == 'NUM' and len(tokens) > 1:
            new_tokens[1] = pluralize(tokens[1])
            new_phrases.append(' '.join(new_tokens))
        return new_phrases

    def get_best_candidate(self, candidate_sents):
        candidate_sents.sort(key=self.score_sent, reverse=True)

        #print([s for s, _, _ in candidate_sents])
        return candidate_sents[0]

    def score_sent(self, candidate):
        sent, _, _ = candidate
        sent = ". "+sent
        try:
            tokens = self.enc.encode(sent)
        except KeyError:
            return 0

        context = torch.tensor(tokens, dtype=torch.long, device = self.device).reshape(1,-1)

        logits, _ = self.model(context)
        log_probs = logits.log_softmax(2)
        sentence_log_prob = 0
        for idx, c in enumerate(tokens):
            if idx > 0:
                sentence_log_prob += log_probs[0, idx-1, c]
        return sentence_log_prob.item() / (len(tokens)**0.2)


class KnowledgeMiner:
    def __init__(self, dev_data_path, device, Template, bert, template_loc = None, language_model = None):
        self.sentences_mask_tail = Template(
            dev_data_path,
            False,
            True,
            device,
            template_loc = template_loc,
            language_model = language_model
        )
        self.sentences_mask_head = Template(
            dev_data_path,
            True,
            False,
            device,
            template_loc = template_loc,
            language_model = language_model
        )
        self.sentences_mask_both = Template(
            dev_data_path,
            True,
            True,
            device,
            template_loc = template_loc,
            language_model = language_model
        )

        bert.eval()
        bert.to(device)
        self.bert = bert

        self.device = device
        self.results = []

    def make_predictions(self):
        data = []
        for idx, ((sent, masked_tail, ids, label), (_, masked_head, _, _), (_, masked_both, _, _)) \
                in enumerate(zip(self.sentences_mask_tail, self.sentences_mask_head, self.sentences_mask_both)):
            tail_masked_ids = [idx for idx, token in enumerate(masked_tail) if token == 103]
            head_masked_ids = [idx for idx, token in enumerate(masked_head) if token == 103]

            # conditional
            logprob_tail_conditional = self.predict(sent, masked_tail, ids, tail_masked_ids)
            logprob_head_conditional = self.predict(sent, masked_head, ids, head_masked_ids)
            # marginal
            logprob_tail_marginal = self.predict(sent, masked_both, ids, tail_masked_ids)
            logprob_head_marginal = self.predict(sent, masked_both, ids, head_masked_ids)

            NLL = -logprob_tail_conditional/len(tail_masked_ids)

            mutual_inf = logprob_tail_conditional - logprob_tail_marginal
            mutual_inf += logprob_head_conditional - logprob_head_marginal
            mutual_inf /= 2.
            try:
                print(idx, (NLL.item(), mutual_inf.item(), label, self.sentences_mask_tail.id_to_text(sent)))
                data.append((NLL.item(), logprob_tail_conditional.item(), logprob_tail_marginal.item(),
                             logprob_head_conditional.item(), logprob_head_marginal.item(),
                             mutual_inf.item(), label, self.sentences_mask_tail.id_to_text(sent)))
            except AttributeError:
                print(idx, (NLL, mutual_inf, label, self.sentences_mask_tail.id_to_text(sent)))
                data.append((NLL,  logprob_tail_conditional.item(), logprob_tail_marginal.item(),
                             logprob_head_conditional.item(), logprob_head_marginal.item(),
                             mutual_inf.item(), label, self.sentences_mask_tail.id_to_text(sent)))
        # prep data
        df = pd.DataFrame(data, columns = ('nll','tail_conditional','tail_marginal',
                                           'head_conditional','head_marginal','mut_inf','label','sent'))
        self.results = df
        return df

    def predict(self, sent, masked, ids, masked_ids):
        logprob = 0
        masked = deepcopy(masked)
        masked_ids = masked_ids.copy()
        for _ in range(len(masked_ids)):
            # make prediction
            pred = self.bert(masked.reshape(1,-1),ids.reshape(1,-1)).log_softmax(2)

            # get log probs for each token
            max_log_prob = -np.inf

            for idx in masked_ids:
                if pred[0, idx, sent[idx]] > max_log_prob:
                    most_likely_idx = idx
                    max_log_prob = pred[0, idx, sent[idx]]

            logprob += max_log_prob
            masked[most_likely_idx] = sent[most_likely_idx]
            masked_ids.remove(most_likely_idx)
            print(self.sentences_mask_tail.id_to_text(pred[0, most_likely_idx,:].topk(10)[1]))
            print(self.sentences_mask_tail.id_to_text(sent[most_likely_idx:most_likely_idx+1]))
        return logprob
