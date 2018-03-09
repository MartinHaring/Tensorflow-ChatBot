import tensorflow as tf
import numpy as np
import data
from datetime import datetime

print('Inference preparation started @ {}'.format(str(datetime.now())))

max_line_length = data.dparams['max_line_length']

print('Initialize vocabulary...')
vocab_to_int, int_to_vocab = data.get_word_dicts()

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

    print("\nEnter 'quit' to end the session.\n")
    infer_time = True

    while infer_time:
        input_question = input('Enter input: ')

        if input_question.lower() == 'quit':
            infer_time = False
            break

        input_question = \
            data.q_to_seq(input_question, vocab_to_int)

        answer_logits = sess.run(logits,
                                 {input_data: [input_question],
                                  keep_prob: 1})[0]

        print('\nTensorBot: {}\n'.format(' '.join([int_to_vocab[i] for i in np.argmax(answer_logits, 1)])))

    print('\n\nSession ended.')

print('\nInference finished @ {}'.format(str(datetime.now())))
