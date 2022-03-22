import numpy as np
import tensorflow as tf


class MLNLP:
    model = None

    def __init__(self):
        pass

    def evaluate(self, text):
        pass

    # Gets shape dynamically to pass to model
    def shape_list(self, x):
        static_shape = x.shape.as_list()
        dynamic_shape = tf.shape(x)
        return [dynamic_shape[i] if s is None else s for i, s in enumerate(static_shape)]
