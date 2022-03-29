import spacy
import pandas as pd


class Tokenizer:

    nlp = None
    vocab = {}

    def __init__(self):
        pass
        self.nlp = spacy.blank("en").from_disk("C:/Users/handw/AppData/Local/Programs/Python/Python37/Lib/site-packages/en_core_web_lg/en_core_web_lg-3.2.0")

    def tokenize(self, text):
        doc = self.nlp(text)

        text, lemma, pos, tag, dep, is_alpha, is_stop = [], [], [], [], [], [], []

        for token in doc:
            text.append(token.text)
            lemma.append(token.lemma_)
            pos.append(token.pos_)
            tag.append(token.tag_)
            dep.append(token.dep_)
            is_alpha.append(token.is_alpha)
            is_stop.append(token.is_stop)

        pd_df = pd.DataFrame({'text': text,
                              'lemma': lemma,
                              'pos': pos,
                              'tag': tag,
                              'dep': dep,
                              'is_alpha': is_alpha,
                              'is_stop': is_stop})

        return pd_df

    # Vocabularize a list of words by adding them to the vocabulary if they do not already exist
    # param add defines whether or not vocab is added or returns -2 for unknown value
    def vocabularize(self, text, add):
        for i, word in enumerate(text):
            word = word.lower()

            # Adds a new word to the vocab list if it does not already exist
            if self.vocab.has_key(word):
                # Replace text with vocab index
                text[i] = self.vocab[word]
            else:
                if add:
                    self.vocab[word] = len(self.vocab)
                else:
                    # Replace text with unknown
                    text[i] = -2

    # Saves the current vocabulary list
    def save_vocab(self):
        pass

    def fill_blanks(self, df, max_entries):
        pass