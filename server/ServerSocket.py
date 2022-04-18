import socket
import utils.DataCleanser as dc


class ServerSocket:

    client, address, socket = None, None, None
    running = False
    model = None

    def __init__(self, model, port):
        # Initialize server socket connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', port))
        print("\nNLPTalk server started!\n")

        self.model = model

        self.running = True

    def listen(self):
        # Listen for client requests
        self.socket.listen()
        self.client, self.address = self.socket.accept()

        while self.running:
            try:
                data = self.client.recv(1024)
                if not data:
                    print("Client sent no data!")
                    print("Restarting listening socket.")
                    self.listen()
                    break
                else:
                    message = data.decode("utf-8")
                    print('Before cleaning: ' + message)
                    # Evaluate with NLP model and send result back to client

                    # Clean up speech message
                    message = dc.clean_speech(message)
                    print('After cleaning: ' + message + '\n')

                    self.client.sendall(str.encode(self.model.evaluate(message)))
                    # self.client.sendall(str.encode('Message received successfully, this is from the server!'))
                    continue
            except socket.timeout:
                print("Socket timed out!")
                print("Restarting listening socket.")
                self.listen()
                break
            except ConnectionResetError:
                print("Socket client has disconnected!")
                print("Restarting listening socket.")
                self.listen()
                break

            except KeyboardInterrupt:
                print("\nNLPTalk server closed with KeyboardInterrupt!\n")
                self.close()
                exit()

    def close(self):
        self.running = False
        try:
            # Close client socket connection
            self.client.close()
            # Close server socket connection
            self.socket.close()
        except Exception as e:
            pass
