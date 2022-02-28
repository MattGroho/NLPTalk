from models.GPT3 import GPT3
from server.ServerSocket import ServerSocket
from utils.InputHandler import listen


def ServerApp():
    # Initialize model
    gpt3 = GPT3()

    ss = ServerSocket(gpt3)
    ss.listen()


# Press the green button to run the script
if __name__ == '__main__':
    ServerApp()
