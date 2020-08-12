import pickle

import os
import re
import tensorflow

if tensorflow.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf
from bert.tokenization import convert_ids_to_tokens

NEG = 0
POS = 1

pattern = re.compile('</?(POS)?(NEG)?>')


def cache_decorator(*dump_fnames):
    def excution_decorator(func):
        def wrapper(*args, **kwargs):
            if len(dump_fnames) == 1:
                dump_fname = dump_fnames[0]
                if not os.path.isfile(dump_fname):
                    ret = func(*args, **kwargs)
                    with open(dump_fname, 'wb') as fdump:
                        pickle.dump(ret, fdump)
                    return ret

                with open(dump_fname, 'rb') as fdump:
                    ret = pickle.load(fdump)
                return ret

            rets = None
            for fname in dump_fnames:
                if not os.path.isfile(fname):
                    rets = func(*args, **kwargs)
                    break
            if rets is not None:
                for r, fname in zip(rets, dump_fnames):
                    with open(fname, 'wb') as fdump:
                        pickle.dump(r, fdump)
                return rets

            rets = []
            for fname in dump_fnames:
                with open(fname, 'rb') as fdump:
                    rets.append(pickle.load(fdump))
            return tuple(rets)

        return wrapper

    return excution_decorator


def convert_ids_to_token_list(input_ids, vocab=None):
    iv_vocab = {input_id: wordpiece for wordpiece, input_id in vocab.items()}
    token_list = convert_ids_to_tokens(iv_vocab, input_ids)
    return token_list


def convert_subtoken_ids_to_tokens(ids, vocab, exps=None, raw_sentence=None):
    subtokens = convert_ids_to_token_list(ids, vocab)
    tokens, exps_outputs = [], []
    if not isinstance(exps[0], list):
        exps = [exps]
    exps_inputs = [[0] * len(ids)] if exps is None else exps
    raw_sentence = subtokens if raw_sentence is None else raw_sentence
    subtokens = list(reversed([t[2:] if t.startswith('##') else t for t in subtokens]))
    exps_inputs = list(zip(*(list(reversed(e)) for e in exps_inputs)))
    for ref_token in raw_sentence:
        t, es = '', [0] * len(exps_inputs[0])
        while t != ref_token and len(subtokens) > 0:
            t += subtokens.pop()
            es = [max(old, new) for old, new in zip(es, exps_inputs.pop())]
        tokens.append(t)
        exps_outputs.append(es)
        if len(subtokens) == 0:
            # the last sub-token is incomplete, ditch it directly
            if ref_token != tokens[-1]:
                tokens = tokens[:-1]
                exps_outputs = exps_outputs[:-1]
            break
    if exps is None:
        return tokens
    return tokens, exps_outputs

