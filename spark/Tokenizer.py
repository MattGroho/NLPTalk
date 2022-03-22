import spacy


class Tokenizer:

    nlp = None
    vocab = {}

    def __init__(self):
        pass
        self.nlp = spacy.blank("en").from_disk("C:/Users/handw/AppData/Local/Programs/Python/Python37/Lib/site-packages/en_core_web_lg/en_core_web_lg-3.2.0")

    def tokenize(self, text):
        doc = self.nlp(text)

        doc_list = []

        for token in doc:
            doc_list.append([token.text, token.lemma_, token.pos, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop])
            print(token.morph)

        return doc_list

    # Vocabularize a list of words by adding them to the vocabulary if they do not already exist
    def vocabularize(self, text):
        for i, word in enumerate(text):
            word = word.lower()

            # Adds a new word to the vocab list if it does not already exist
            if not self.vocab.has_key(word):
                self.vocab[word] = len(self.vocab)

            # Replace text with vocab index
            text[i] = self.vocab[word]

    # Saves the current vocabulary list
    def save_vocab(self):
        pass
