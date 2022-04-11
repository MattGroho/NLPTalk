from spark.Tokenizer import Tokenizer
from models.GPT3 import GPT3
from models.GPTNeo import GPTNeo
#from utils.InputHandler import listen
from utils import DataCleanser as dc

DO_VOICE_INPUT = False


def main():
    doc_list = [line for line in dc.read_doc('data/EtownDocData.txt').split('\n') if line]

    combined_doc_list = [doc_list[i:i+14] for i in range(0, len(doc_list), 14)]

    final_doc_list = []

    for group in range(len(combined_doc_list)):
        final_doc_list.append('')
        for element in combined_doc_list[group]:
           final_doc_list[group] += element

    #final_doc_list = [final_doc_list[len(final_doc_list) - 1]]

    # Initialize model
    gpt3 = GPT3('ETOWN', final_doc_list)
    # gptNeo = GPTNeo('2.7B')

    print(gpt3.evaluate('Who founded Elizabethtown College?'))


# Press the green button to run the script
if __name__ == '__main__':
    main()
