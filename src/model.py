import tensorflow as tf


# Create placeholders for inputs to the model, which are initially empty
def model_inputs():

    input_data = \
        tf.placeholder(tf.int32,
                       [None, None],
                       name='input')

    targets = \
        tf.placeholder(tf.int32,
                       [None, None],
                       name='targets')

    lr = \
        tf.placeholder(tf.float32,
                       name='learning_rate')

    keep_prob = \
        tf.placeholder(tf.float32,
                       name='keep_prob')

    return input_data, targets, lr, keep_prob


# Remove last word id from each batch and concat <GO> to the start
def process_encoding_input(target_data,
                           vocab_to_int,
                           batch_size):

    sliced_data = \
        tf.strided_slice(target_data,
                         [0, 0],
                         [batch_size, -1],
                         [1, 1])

    dec_input = \
        tf.concat([tf.fill([batch_size, 1],
                           vocab_to_int['<GO>']),
                   sliced_data], 1)

    return dec_input


# Create the encoding layer
def encoding_layer(rnn_inputs,
                   rnn_size,
                   num_layers,
                   keep_prob,
                   seq_length,
                   attn_length):

    lstm = \
        tf.contrib.rnn.BasicLSTMCell(rnn_size)

    drop = \
        tf.contrib.rnn.DropoutWrapper(lstm,
                                      input_keep_prob=keep_prob)

    cell = tf.contrib.rnn.AttentionCellWrapper(drop,
                                               attn_length,
                                               state_is_tuple=True)

    enc_cell = \
        tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

    _, enc_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell,
                                        cell_bw=enc_cell,
                                        sequence_length=seq_length,
                                        inputs=rnn_inputs,
                                        dtype=tf.float32)

    return enc_state


# Decode training data
def decoding_layer_train(encoder_state,
                         dec_cell,
                         dec_embed_input,
                         sequence_length,
                         decoding_scope,
                         output_fn,
                         keep_prob):

    train_decoder_fn = \
        tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)

    train_pred, _, _ = \
        tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                               train_decoder_fn,
                                               dec_embed_input,
                                               sequence_length,
                                               scope=decoding_scope)

    train_pred_drop = \
        tf.nn.dropout(train_pred,
                      keep_prob)

    return output_fn(train_pred_drop)


# Decode prediction data
def decoding_layer_infer(encoder_state,
                         dec_cell,
                         dec_embeddings,
                         start_of_sequence_id,
                         end_of_sequence_id,
                         maximum_length,
                         vocab_size,
                         decoding_scope,
                         output_fn):

    infer_decoder_fn = \
        tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn,
                                                       encoder_state,
                                                       dec_embeddings,
                                                       start_of_sequence_id,
                                                       end_of_sequence_id,
                                                       maximum_length,
                                                       vocab_size)

    infer_logits, _, _ = \
        tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                               infer_decoder_fn,
                                               scope=decoding_scope)

    return infer_logits


# Create decoding cell and input parameters for training and inference decoding layers
def decoding_layer(dec_embed_input,
                   dec_embeddings,
                   encoder_state,
                   vocab_size,
                   sequence_length,
                   rnn_size,
                   num_layers,
                   vocab_to_int,
                   keep_prob,
                   attn_length):

    with tf.variable_scope('decoding') as decoding_scope:

        lstm = \
            tf.contrib.rnn.BasicLSTMCell(rnn_size)

        drop = \
            tf.contrib.rnn.DropoutWrapper(lstm,
                                          output_keep_prob=keep_prob)

        cell = \
            tf.contrib.rnn.AttentionCellWrapper(drop,
                                                attn_length,
                                                state_is_tuple=True)

        dec_cell = \
            tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        weights = \
            tf.truncated_normal_initializer(stddev=0.1)

        biases = \
            tf.zeros_initializer()

        output_fn = \
            lambda x: \
            tf.contrib.layers.fully_connected(x,
                                              vocab_size,
                                              None,
                                              scope=decoding_scope,
                                              weights_initializer=weights,
                                              biases_initializer=biases)

        train_logits = \
            decoding_layer_train(encoder_state[0],
                                 dec_cell,
                                 dec_embed_input,
                                 sequence_length,
                                 decoding_scope,
                                 output_fn,
                                 keep_prob)

        decoding_scope.reuse_variables()

        infer_logits = \
            decoding_layer_infer(encoder_state[0],
                                 dec_cell,
                                 dec_embeddings,
                                 vocab_to_int['<GO>'],
                                 vocab_to_int['<EOS>'],
                                 sequence_length,
                                 vocab_size,
                                 decoding_scope,
                                 output_fn)

        return train_logits, infer_logits


# Use previous functions to create training and inference logits
def seq2seq_model(input_data,
                  target_data,
                  keep_prob,
                  batch_size,
                  sequence_length,
                  vocab_size,
                  enc_embedding_size,
                  dec_embedding_size,
                  rnn_size,
                  num_layers,
                  vocab_to_int,
                  attn_length):

    enc_embed_input = \
        tf.contrib.layers.embed_sequence(input_data,
                                         vocab_size + 1,
                                         enc_embedding_size)

    enc_state = \
        encoding_layer(enc_embed_input,
                       rnn_size,
                       num_layers,
                       keep_prob,
                       sequence_length,
                       attn_length)

    dec_input = \
        process_encoding_input(target_data,
                               vocab_to_int,
                               batch_size)

    dec_embeddings = \
        tf.Variable(tf.random_uniform([vocab_size + 1,
                                       dec_embedding_size], -1, 1))

    dec_embed_input = \
        tf.nn.embedding_lookup(dec_embeddings,
                               dec_input)

    train_logits, infer_logits = \
        decoding_layer(dec_embed_input,
                       dec_embeddings,
                       enc_state,
                       vocab_size + 1,
                       sequence_length,
                       rnn_size,
                       num_layers,
                       vocab_to_int,
                       keep_prob,
                       attn_length)

    return train_logits, infer_logits
