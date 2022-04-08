from spark.Tokenizer import Tokenizer
from models.GPT3 import GPT3
from models.GPTNeo import GPTNeo
#from utils.InputHandler import listen
from utils import DataCleanser as dc

DO_VOICE_INPUT = False


def main():
    # Initialize model
    gpt3 = GPT3('ETOWN', [line for line in dc.read_doc('EtownData.txt').split('\n') if line])
    # gptNeo = GPTNeo('2.7B')

    print(gpt3.evaluate('When was etown founded?'))


# Press the green button to run the script
if __name__ == '__main__':
    main()
