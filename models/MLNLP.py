import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model


class MLNLP:
    model = None

    def __init__(self, input_shape,
                 n_classes,
                 filters=250,
                 kernel_size=3,
                 strides=1,
                 dense_units=128,
                 dropout_rate=0.,
                 CNN_layers=2,
                 clf_reg=1e-4):
        # Model Definition
        # raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)
        xcnn = Conv2D(filters,
                      (kernel_size),
                      padding='same',
                      activation='relu',
                      strides=strides,
                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                      name='Conv2D_1')(raw_inputs)

        xcnn = BatchNormalization()(xcnn)
        xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv2D(filters,
                          (kernel_size),
                          padding='same',
                          activation='relu',
                          strides=strides,
                          kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                          bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                          activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                          name='Conv2D_' + str(i + 1))(xcnn)

            xcnn = BatchNormalization()(xcnn)
            xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

            if dropout_rate != 0:
                xcnn = Dropout(dropout_rate)(xcnn)

                # we flatten for dense layer
        xcnn = Flatten()(xcnn)

        xcnn = Dense(dense_units, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                     bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                     activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                     name='FC1_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        xcnn = Dense(dense_units, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                     bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                     activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                     name='FC2_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        top_level_predictions = Dense(n_classes, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                      name='top_level_output')(xcnn)

        # [512, 340, 20, 10, 5, 520, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)
        print(self.model.summary())  # summarize layers
        plot_model(self.model, to_file=save_dir + '/model.png')  # plot graph
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                           metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train, one_hot(y_train, self.n_classes),
                              batch_size=n_batch,
                              epochs=n_epochs,
                              validation_data=(X_val, one_hot(y_val, self.n_classes)))

    def classify(self, data):
        if len(data.shape) > 2:
            return self.model.predict(data.reshape(-1, data.shape[1], data.shape[2], 1))
        else:
            return self.model.predict(data)
