import os
from transformers.tokenization_bert import WordpieceTokenizer, BertTokenizer, BasicTokenizer, whitespace_tokenize
from transformers.file_utils import http_get


def printable_text(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class BasicRationalTokenizer(BasicTokenizer):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def _is_token_rational(self, token_idx, evidences):
        if evidences is None:
            return 0
        for ev in evidences:
            if token_idx >= ev.start_token and token_idx < ev.end_token:
                return 1
        return 0

    def tokenize(self, text, evidences):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        split_rations = []
        for token_idx, token in enumerate(orig_tokens):
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            sub_tokens = self._run_split_on_punc(token)
            sub_tokens = ' '.join(sub_tokens).strip().split()
            if len(sub_tokens) > 0:
                split_tokens.extend(sub_tokens)
                ration = self._is_token_rational(token_idx, evidences)
                split_rations.extend([ration] * len(sub_tokens))
        return zip(split_tokens, split_rations)


class FullRationaleTokenizer(BertTokenizer):  # Test passed :)
    def __init__(self, do_lower_case=True):
        if not os.path.isfile('bert-base-uncased-vocab.txt'):
            http_get("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
                     open("bert-base-uncased-vocab.txt", 'wb+'))
        super(FullRationaleTokenizer, self).__init__('bert-base-uncased-vocab.txt')
        self.basic_rational_tokenizer = BasicRationalTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token="[UNK]")

    def tokenize(self, text, evidences=None):
        split_tokens = []
        split_rations = []
        for token, ration in self.basic_rational_tokenizer.tokenize(text, evidences):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
                split_rations.append(ration)
        return list(zip(split_tokens, split_rations))


