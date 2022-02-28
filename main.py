from models.GPT3 import GPT3
from models.GPTNeo import GPTNeo
from utils.InputHandler import listen

DO_VOICE_INPUT = False


def main():
    # Initialize model
    gpt3 = GPT3()
    gptNeo = GPTNeo('2.7B')

    while True:
        # Listen to user input and speak back response
        listen(gptNeo, DO_VOICE_INPUT)


# Press the green button to run the script
if __name__ == '__main__':
    main()
