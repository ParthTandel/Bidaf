
from tensorflow.keras import backend as K

def calculate_accuracy(true_and_pred):
    y_true, y_pred = true_and_pred
    start_prob = y_pred[0][K.cast(y_true[0], dtype='int32')]
    end_prob = y_pred[1][K.cast(y_true[1], dtype='int32')]
    return (start_prob + end_prob) / 2.0

def log_probabilities(true_and_pred):
    y_true, y_pred = true_and_pred
    start_probability = y_pred[0][K.cast(y_true[0], dtype='int32')]
    end_probability = y_pred[1][K.cast(y_true[1], dtype='int32')]
    return K.log(start_probability) + K.log(end_probability)

def accuracy(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=1)
    return K.mean(K.map_fn(calculate_accuracy, (y_true, y_pred), dtype='float32'), axis=0)

def negative_log_prob(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=1)
    return -K.mean(K.map_fn(log_probabilities, (y_true, y_pred), dtype='float32'), axis=0)
