import pickle
import socket
import utils.DataCleanser as dc

from models.GPT3 import GPT3
from models.GPTNeo import GPTNeo
from models.LDA import LDA
from models.Siamese import Siamese


class ServerSocket:

    client, address, socket = None, None, None
    running = False
    model = None

    use_encoded_responses = False
    prevModel = -1

    def __init__(self, port):
        # Initialize server socket connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', port))
        print("\nNLPTalk server started!\n")

        self.running = True

    def listen(self):
        # Listen for client requests
        self.socket.listen()
        self.client, self.address = self.socket.accept()
        self.socket.settimeout(5.0)

        while self.running:
            try:
                data = self.client.recv(1024)
                if not data:
                    print("Client sent no data!")
                    print("Restarting listening socket.")
                    self.close()
                    break
                else:
                    message = data.decode("utf-8").split('|')

                    model_select = int(message[0])
                    message = message[1]

                    print('Before cleaning: ' + message)
                    # Evaluate with NLP model and send result back to client

                    # Clean up speech message
                    message = dc.clean_speech(message)
                    print('After cleaning: ' + message + '\n')

                    if dc.is_junk(message):
                        self.client.sendall(str.encode('Sorry, I did not understand that. Could you rephrase your question?'))
                        continue

                    if self.prevModel != model_select:
                        # SNN
                        if model_select == 0:
                            self.use_encoded_responses = True
                            self.model = Siamese()
                        # LDA
                        elif model_select == 1:
                            self.use_encoded_responses = True
                            self.model = LDA()
                            pass
                        # GPT-3
                        elif model_select == 2:
                            self.use_encoded_responses = False
                            self.model = GPT3('GPT3', None)
                        # GPT-3-Etown
                        elif model_select == 3:
                            self.use_encoded_responses = False
                            final_doc_list = pickle.load(open('data/EtownQAData.pkl', 'rb'))
                            self.model = GPT3('ETOWN', final_doc_list)
                        # GPT-Neo
                        elif model_select == 4:
                            self.use_encoded_responses = False
                            self.model = GPTNeo('2.7B')
                        # WIP Model Types
                        elif model_select >= 5:
                            self.use_encoded_responses = False
                            continue

                    if self.use_encoded_responses:
                        self.client.sendall(str.encode(dc.encode_response(self.model.evaluate(message))))
                    else:
                        self.client.sendall(str.encode(self.model.evaluate(message)))
                    continue
            except socket.timeout:
                print("Socket timed out!")
                print("Restarting listening socket.")
                self.close()
                break
            except ConnectionResetError:
                print("Socket client has disconnected!")
                print("Restarting listening socket.")
                self.close()
                break

            except KeyboardInterrupt:
                print("\nNLPTalk server closed with KeyboardInterrupt!\n")
                self.close()
                exit(0)

            except Exception as e:
                print(e)
                self.close()
                break

    def close(self):
        self.running = False
        try:
            # Close client socket connection
            self.client.close()
            # Close server socket connection
            self.socket.close()
        except Exception as e:
            pass
