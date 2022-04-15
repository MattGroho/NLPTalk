import docx2txt


def save_doc_as_txt(doc_loc, new_doc_loc):
    text = docx2txt.process(doc_loc)

    with open(new_doc_loc, 'wb') as text_file:
        print(text, file=text_file)


def read_doc(doc_loc):
    with open(doc_loc, 'rb') as text_file:
        return text_file.read()

    return None
