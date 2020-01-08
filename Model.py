
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import plot_model

import numpy as np
import json


def BuildModel(num_context_token, num_query_token, num_hidden_state = 200):
    # context and query layer

    query_inputs = Input(shape=(num_query_token, ))
    query_embedding = Embedding(input_dim=num_query_token, output_dim=num_hidden_state, mask_zero=True)(query_inputs)
    query_lstm = Bidirectional(LSTM(num_hidden_state, return_sequences = True))
    query_output = query_lstm(query_embedding)

    context_inputs = Input(shape=(num_context_token, ))
    context_embedding = Embedding(input_dim=num_context_token, output_dim=num_hidden_state, mask_zero=True)(context_inputs)
    context_lstm = Bidirectional(LSTM(num_hidden_state, return_sequences=True))
    context_output = context_lstm(context_embedding)


    # similarity layer

    repeated_context_vectors = K.tile(K.expand_dims(context_output, axis=2), [1, 1, num_query_token, 1])
    repeated_query_vectors = K.tile(K.expand_dims(query_output, axis=1), [1, num_context_token, 1, 1])
    element_wise_mul = repeated_query_vectors * repeated_context_vectors
    concatenated_tensor = K.concatenate([repeated_context_vectors, repeated_query_vectors, element_wise_mul], axis=-1)
    similarity_output = Dense(1)(concatenated_tensor)
    similarity_output = K.squeeze(similarity_output, -1)  # (batch_size, total_passage_words, max_question_words)


    # context to query attention

    query_attention = softmax(similarity_output, axis=-1)
    query_attention = K.tile(K.expand_dims(query_attention, axis=-1), [1, 1, 1, num_hidden_state * 2])
    cont_to_query_attn = query_attention * repeated_query_vectors
    cont_to_query_attn =  K.sum(cont_to_query_attn, axis=-2)


    # query to context attention

    context_attention = softmax(K.max(similarity_output, axis=-1))
    query_to_cont_attn = K.sum((K.expand_dims(context_attention, axis=-1) * context_output), axis=-2)
    query_to_cont_attn = K.tile(K.expand_dims(query_to_cont_attn, axis=-2), [1, num_context_token, 1])


    # combining context, cont_to_query_attn, query_to_cont_attn

    merged_context = K.concatenate([context_output, cont_to_query_attn, 
        context_output * cont_to_query_attn, context_output * query_to_cont_attn], axis=-1)

    # modeling layer

    modeling_lstm_1 = Bidirectional(LSTM(num_hidden_state, return_sequences=True))
    modeling_output_1 = modeling_lstm_1(merged_context)
    modeling_lstm_2 = Bidirectional(LSTM(num_hidden_state, return_sequences=True))
    modeling_output_2 = modeling_lstm_2(modeling_output_1)

    # begin span output layer

    begin_span_merge = K.concatenate([merged_context, modeling_output_2], axis=-1)
    begin_span_layer = TimeDistributed(Dense(1))
    begin_span = begin_span_layer(begin_span_merge)
    begin_span =  K.squeeze(begin_span, axis=-1)
    begin_span = softmax(begin_span)

    # end span output layer

    modeling_lstm_3 =  Bidirectional(LSTM(num_hidden_state, return_sequences=True))
    modeling_output_3 = modeling_lstm_3(modeling_output_2)
    end_span_merge = K.concatenate([merged_context, modeling_output_3], axis=-1)
    end_span_layer = TimeDistributed(Dense(1))
    end_span = end_span_layer(end_span_merge)
    end_span =  K.squeeze(end_span, axis=-1)
    end_span = softmax(end_span)


    model = Model([query_inputs, context_inputs],[begin_span, end_span])
    model.summary()

    print(begin_span.shape, end_span.shape)

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


if __name__ == "__main__":
    num_query_token = 10
    num_context_token = 20
    num_hidden_state = 200

    model = BuildModel(num_context_token, num_query_token, num_hidden_state)
