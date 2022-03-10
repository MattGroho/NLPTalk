import spacy


class Tokenizer:

    nlp = None

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def tokenize(self, text):
        doc = self.nlp(text)

        doc_list = []

        for token in doc:
            doc_list.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop])

        return doc_list
