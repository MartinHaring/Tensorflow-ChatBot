import tensorflow as tf
import numpy as np
import data
# from tensorflow.contrib.seq2seq import sequence_loss
from datetime import datetime
# from model import model_inputs, seq2seq_model

print('Inference preparation started @ {}'.format(str(datetime.now())))
print('Initialize vocabulary...')
vocab_to_int, int_to_vocab = data.get_word_dicts()

'''     --- Re-initialize graph to clear tf.sess ---
print('Initialize training set...')
sorted_questions, sorted_answers = data.get_sorted_qa()

print('Initialize graph...')
train_graph = tf.Graph()

with train_graph.as_default():

    print('Initialize model...')
    # Create placeholders for inputs to the model, which are initially empty
    input_data, targets, lr, keep_prob = model_inputs()

    # Sequence length will be the max line length for each batch
    sequence_length = \
        tf.placeholder_with_default(data.dparams['max_line_length'],
                                    None,
                                    name='sequence_length')

    # Find the shape of the input data for sequence_loss
    input_shape = tf.shape(input_data)

    # Create training and inference logits
    train_logits, inference_logits = \
        seq2seq_model(tf.reverse(input_data, [-1]),
                      targets,
                      keep_prob,
                      data.hparams['batch_size'],
                      sequence_length,
                      len(vocab_to_int),
                      data.hparams['encoding_embedding_size'],
                      data.hparams['decoding_embedding_size'],
                      data.hparams['rnn_size'],
                      data.hparams['num_layers'],
                      vocab_to_int,
                      data.hparams['attn_length'])

    # Create a tensor to be used for making predictions
    tf.identity(inference_logits, 'logits')

    print('Optimize RNN...')
    with tf.name_scope('optimization'):

        # Loss function
        cost = \
            sequence_loss(train_logits,
                          targets,
                          tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = \
            tf.train.AdamOptimizer(data.hparams['learning_rate'])

        # Gradient Clipping
        gradients = \
            optimizer.compute_gradients(cost)

        capped_gradients = \
            [(tf.clip_by_value(grad, -1., 1.), var)
             for grad, var in gradients
             if grad is not None]

        train_op = \
            optimizer.apply_gradients(capped_gradients)


print('Initialize training parameters...')
# Validate the training with 10% of the data
train_valid_split = \
    int(len(sorted_questions)*0.1)

# Split questions and answers into training and validating data
train_questions = \
    sorted_questions[train_valid_split:]

train_answers = \
    sorted_answers[train_valid_split:]

valid_questions = \
    sorted_questions[:train_valid_split]

valid_answers = \
    sorted_answers[:train_valid_split]

# Record training loss for each display step
total_train_loss = 0

# Record validation loss for saving improvements in the model
summary_valid_loss = []

# Variable to keep track of consecutive validation losses
stop_early = 0

# Modulus for checking validation loss (check when 50% and 100% are done)
validation_check = \
    ((len(train_questions)) // data.hparams['batch_size'] // 2) - 1


print('Training preparation finished @ {}\n'.format(str(datetime.now())))

print('Start session...')
with tf.Session(graph=train_graph) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Check if checkpoint exists...')
    if tf.train.checkpoint_exists(data.tparams['checkpoint']):
        print('Load checkpoint...')
        saver.restore(sess, data.tparams['checkpoint'])
'''

print('Initialize graph...')
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:

    print('Load model and tensors...')
    loader = tf.train.import_meta_graph(data.tparams['checkpoint'] + '.meta')
    loader.restore(sess, data.tparams['checkpoint'])

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    print('Inference preparation finished @ {}\n'.format(str(datetime.now())))
    print('\nInference started @ {}'.format(str(datetime.now())))
    print("(Enter 'quit' to end the session)")
    while True:
        input_question = input('\nEnter input: ')

        if input_question.lower() == 'quit':
            break

        input_question = \
            data.q_to_seq(input_question, vocab_to_int)

        answer_logits = sess.run(logits,
                                 {input_data: [input_question],
                                  keep_prob: 1})[0]

        print('\nTensorBot: {}'.format(' '.join([int_to_vocab[i] for i in np.argmax(answer_logits, 1)])))
    print('\n\nSession ended.')
print('\nInference finished @ {}'.format(str(datetime.now())))
