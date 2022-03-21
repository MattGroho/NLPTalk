import spacy


class Tokenizer:

    nlp = None

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
