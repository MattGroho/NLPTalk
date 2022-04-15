from models.GPT3 import GPT3
from server.ServerSocket import ServerSocket
from utils import DataCleanser as dc


def ServerApp():
    # Create doc_list to pass into GPT model
    doc_list = [line for line in str(dc.read_doc('data/EtownDocData.txt'), 'ISO-8859-1').split('\n') if line]

    combined_doc_list = [doc_list[i:i+14] for i in range(0, len(doc_list), 14)]

    final_doc_list = []

    for group in range(len(combined_doc_list)):
        final_doc_list.append('')
        for element in combined_doc_list[group]:
           final_doc_list[group] += element

    # Initialize model
    gpt3 = GPT3('ETOWN', final_doc_list)

    ss = ServerSocket(gpt3, 4824)
    ss.listen()


# Press the green button to run the script
if __name__ == '__main__':
    ServerApp()
