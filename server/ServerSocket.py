import socket


class ServerSocket:

    client, address, socket = None, None, None

    model = None

    def __init__(self, model, port):
        # Initialize server socket connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
            self.close()

    def close(self):
        print("Closing Server Socket")
        try:
            # Close client socket connection
            self.client.close()
            # Close server socket connection
            self.socket.close()
        except Exception as e:
            print('An error has occurred')
