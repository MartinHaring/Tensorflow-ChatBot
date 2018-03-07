import tensorflow as tf
import numpy as np
import data
from tensorflow.contrib.seq2seq import sequence_loss
from datetime import datetime
from model import model_inputs, seq2seq_model

print('Inference preparation started @ {}'.format(str(datetime.now())))

input_question = ''
max_line_length = data.dparams['max_line_length']

print('Initialize vocabulary...')
vocab_to_int, int_to_vocab = data.get_word_dicts()

tf.reset_default_graph()
sess = tf.Session()

print('Initialize graph...')
# Reset the graph to ensure that it is ready for training
train_graph = tf.Graph()

with train_graph.as_default():

    print('Initialize model...')
    input_data, targets, lr, keep_prob = model_inputs()

    sequence_length = \
        tf.placeholder_with_default(max_line_length,
                                    None,
                                    name='sequence_length')

    # Find the shape of the input data for sequence_loss
    input_shape = tf.shape(input_data)

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


print('Load neural network...')
tf.identity(inference_logits, 'logits')

loader = tf.train.import_meta_graph(data.tparams['checkpoint'] + '.meta')
loader.restore(sess, data.tparams['checkpoint'])


print('Inference preparation finished @ {}\n'.format(str(datetime.now())))


while input_question != 'quit':
    input_question = input('Enter input: ')

    if input_question == 'quit':
        break

    # Prepare question
    input_question = \
        data.q_to_seq(input_question, vocab_to_int)

    # Pad the question until it equals the max line length
#    input_question = \
#        input_question + [vocab_to_int['<PAD>']] * (max_line_length - len(input_question))

    # Add empty questions so input data is correct shape
#    batch_shell = \
#        np.zeros((data.hparams['batch_size'], max_line_length))

    # Set first question to be out input question
#    answer_logits = \
#        sess.run(inference_logits,
#                 {input_data: batch_shell,
#                  keep_prob: 1.0})[0]

    # Remove padding from Question and Answer
#    pad_q = vocab_to_int['<PAD>']
#    pad_a = vocab_to_int['<PAD>']

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load the saved model
        loader = tf.train.import_meta_graph(data.tparams['checkpoint'] + '.meta')
        loader.restore(sess, data.tparams['checkpoint'])

        # Load the tensors to be used as inputs
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('logits:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        answer_logits = sess.run(logits,
                                 {input_data: [input_question],
                                  keep_prob: 1.0})[0]

#    print('Question')
#    print('  Word Ids:       {}'.format([i for i in input_question if i != pad_q]))
#    print('  Input Words:    {}'.format([int_to_vocab[i] for i in input_question if i != pad_q]))

#    print('\nAnswer')
#    print('  Word Ids:       {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
#    print('  Response Words: {}'.format([int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))

    print('Question')
    print('  Word Ids:       {}'.format([i for i in input_question]))
    print('  Input Words:    {}'.format([int_to_vocab[i] for i in input_question]))

    print('\nAnswer')
    print('  Word Ids:       {}'.format([i for i in np.argmax(answer_logits, 1)]))
    print('  Response Words: {}'.format([int_to_vocab[i] for i in np.argmax(answer_logits, 1)]))

print('Session ended.')
