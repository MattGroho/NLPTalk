from models.GPT3 import GPT3
from server.ServerSocket import ServerSocket
from utils.voice import listen


def main():
    # Initialize model
    gpt3 = GPT3()

    ss = ServerSocket(gpt3)
    ss.listen()

    #while True:
        # Listen to user input and speak back response
        #listen(gpt3)


# Press the green button to run the script
if __name__ == '__main__':
    main()
