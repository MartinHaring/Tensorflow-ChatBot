import tensorflow as tf
import numpy as np
import data
import training
from model import seq2seq_model


print('Initialize Session...')
input_question = ''

vocab_to_int, int_to_vocab = data.get_word_dicts()
max_line_length = data.params['max_line_length']

tf.reset_default_graph()
sess = tf.Session()

input_data = \
    tf.placeholder(tf.int32,
                   [None, None],
                   name='input')

targets = \
    tf.placeholder(tf.int32,
                   [None, None],
                   name='targets')

keep_prob = \
    tf.placeholder(tf.float32,
                   name='keep_prob')

sequence_length = \
    tf.placeholder_with_default(max_line_length,
                                None,
                                name='sequence_length')

train_logits, inference_logits = \
    seq2seq_model(tf.reverse(input_data, [-1]),
                  targets,
                  keep_prob,
                  training.hparams['batch_size'],
                  sequence_length,
                  len(vocab_to_int),
                  training.hparams['encoding_embedding_size'],
                  training.hparams['decoding_embedding_size'],
                  training.hparams['rnn_size'],
                  training.hparams['num_layers'],
                  vocab_to_int)

# Create a tensor for inference logits, needed for loading checkpoints
tf.identity(inference_logits, 'logits')

saver = tf.train.import_meta_graph(training.tparams['checkpoint'] + '.meta')
saver.restore(sess, training.tparams['checkpoint'])

while input_question != 'quit':
    input_question = input('Enter input: ')

    # Prepare question
    input_question = \
        data.q_to_seq(input_question, vocab_to_int)

    # Pad the question until it equals the max line length
    input_question = \
        input_question + [vocab_to_int['<PAD>']] * (max_line_length - len(input_question))

    # Add empty questions so input data is correct shape
    batch_shell = \
        np.zeros((training.hparams['batch_size'], max_line_length))

    # Set first question to be out input question
    answer_logits = \
        sess.run(inference_logits,
                 {input_data: batch_shell,
                  keep_prob: 1.0})[0]

    # Remove padding from Question and Answer
    pad_q = vocab_to_int['<PAD>']
    pad_a = vocab_to_int['<PAD>']

    print('Question')
    print('  Word Ids:       {}'.format([i for i in input_question if i != pad_q]))
    print('  Input Words:    {}'.format([int_to_vocab[i] for i in input_question if i != pad_q]))

    print('\nAnswer')
    print('  Word Ids:       {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
    print('  Response Words: {}'.format([int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))

print('Session ended.')
