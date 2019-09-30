from copy import deepcopy
import pandas as pd
import numpy as np

class KnowledgeMiner:

    def __init__(self, dev_data_path, device, Template, bert, **kwarg):
        """ Creates a class instance for doing KBC with a given template and
        HuggingFace bert model. Template classes defined in `sentences.py` """
        self.sentences = Template(
            dev_data_path,
            device,
            **kwarg
        )

        bert.eval()
        bert.to(device)
        self.bert = bert

        self.device = device
        self.results = []

    def make_predictions(self):
        data = []
        for idx, (sent, (masked_head, masked_tail, masked_both), ids, label) in enumerate(self.sentences):
            tail_masked_ids = [idx for idx, token in enumerate(masked_tail) if token == 103]
            head_masked_ids = [idx for idx, token in enumerate(masked_head) if token == 103]

            # conditional
            logprob_tail_conditional = self.predict(sent, masked_tail, ids, tail_masked_ids)
            logprob_head_conditional = self.predict(sent, masked_head, ids, head_masked_ids)
            # marginal
            logprob_tail_marginal = self.predict(sent, masked_both, ids, tail_masked_ids)
            logprob_head_marginal = self.predict(sent, masked_both, ids, head_masked_ids)

            NLL = -logprob_tail_conditional/len(tail_masked_ids)

            # average approximations of PMI(t,h|r) and PMI(h,t|r)
            mutual_inf = logprob_tail_conditional - logprob_tail_marginal
            mutual_inf += logprob_head_conditional - logprob_head_marginal
            mutual_inf /= 2.

            try:
                print(idx, (NLL.item(), mutual_inf.item(), label, self.sentences.id_to_text(sent)))
                data.append((NLL.item(), logprob_tail_conditional.item(), logprob_tail_marginal.item(),
                             logprob_head_conditional.item(), logprob_head_marginal.item(),
                             mutual_inf.item(), label, self.sentences.id_to_text(sent)))
            except AttributeError:
                print(idx, (NLL, mutual_inf, label, self.sentences.id_to_text(sent)))
                data.append((NLL,  logprob_tail_conditional.item(), logprob_tail_marginal.item(),
                             logprob_head_conditional.item(), logprob_head_marginal.item(),
                             mutual_inf.item(), label, self.sentences.id_to_text(sent)))

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

        return logprob
