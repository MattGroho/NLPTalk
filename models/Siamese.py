import tensorflow as tf

from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, LSTM, Embedding
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.python.keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D
from tensorflow.python.keras import backend as K



class Siamese:
    model, embedding_index, embeddings_matrix = None, None, None

    def __init__(self, df_pkl, embeddings):
        t = Tokenizer()

        """
        Modified Siamese Network for computing similar sentence classification
        Original code snippets courtesy of Prabhnoor Singh
        https://github.com/prabhnoor0212
        """

        input_1 = Input(shape=(df_pkl.train_q1_seq.shape[1],))
        input_2 = Input(shape=(df_pkl.train_q2_seq.shape[1],))

        # Import weights from word embeddings - make them non-trainable
        common_embed = Embedding(name="synopsis_embedd", input_dim=len(t.word_index) + 1,
                                 output_dim=len(self.embeddings_index['no']), weights=[self.embedding_matrix],
                                 input_length=df_pkl.train_q1_seq.shape[1], trainable=False)
        lstm_1 = common_embed(input_1)
        lstm_2 = common_embed(input_2)

        """ Beginning of data processing structure """

        common_lstm = LSTM(64, return_sequences=True, activation="relu")
        vector_1 = common_lstm(lstm_1)
        vector_1 = Flatten()(vector_1)

        vector_2 = common_lstm(lstm_2)
        vector_2 = Flatten()(vector_2)

        x3 = Subtract()([vector_1, vector_2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([vector_1, vector_1])
        x2_ = Multiply()([vector_2, vector_2])
        x4 = Subtract()([x1_, x2_])

        # https://stackoverflow.com/a/51003359/10650182
        x5 = Lambda(self.cosine_distance, output_shape=self.cos_dist_output_shape)([vector_1, vector_2])

        # Assemble vector containing [cosine similarity, ]
        conc = Concatenate(axis=-1)([x5, x4, x3])

        """ End of data processing structure """

        # Add fully connected layer
        x = Dense(100, activation="relu", name='conc_layer')(conc)

        # Account for overfitting
        x = Dropout(0.01)(x)

        # Obtain similarity result (1 if matching class, 0 otherwise)
        out = Dense(1, activation="sigmoid", name='out')(x)

        # Save the model and compile
        model = Model([input_1, input_2], out)
        model.compile(loss="binary_crossentropy", metrics=['acc', self.auroc], optimizer=Adam(0.00001))

    def prepareEmbeddings(self):
        pass

    def train(self, df_pkl):
        self.model.fit([df_pkl.train_q1_seq, df_pkl.train_q2_seq], df_pkl.y_train.values.reshape(-1, 1), epochs=5,
                  batch_size=64, validation_data=([df_pkl.test_q1_seq, df_pkl.test_q2_seq], df_pkl.y_test.values.reshape(-1, 1)))

    def classify(self, data):
        pass

    @staticmethod
    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    @staticmethod
    def cosine_distance(vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    @staticmethod
    def cos_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
