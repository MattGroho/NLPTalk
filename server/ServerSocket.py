import socket


class ServerSocket:

    client, address, socket = None, None, None

    model = None

    def __init__(self, model, port):
        # Initialize server socket connection
        self.socket = self.socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', port))

        self.model = model

    def listen(self):
        # Listen for client requests
        self.socket.listen()

        try:
            self.client, self.address = self.socket.accept()

            while 1:
                data = self.client.recv(1024)
                if data:
                    # Evaluate with NLP model and send result back to client
                    self.client.sendall(self.model.evaluate(data))

        finally:
            print("Closing Server Socket")
            self.client.close()

    def close(self):
        # Close server socket connection
        self.socket.close()
