import tensorflow as tf
import numpy as np
import time
import data
from tensorflow.contrib.seq2seq import sequence_loss
from datetime import datetime
from model import model_inputs, seq2seq_model

# ---------- Preparations ----------
print('Training preparation started @ {}'.format(str(datetime.now())))
print('Initialize params...')
max_length = data.dparams['max_line_length']

rnn_size, num_layers, enc_embed_size, dec_embed_size, attn_length, lr, lr_decay, min_lr, keep_probability = \
    data.fetch_hparams()

epochs, batch_size, display_step, stop, checkpoint = \
    data.fetch_tparams()

print('Initialize vocabulary...')
vocab_to_int, int_to_vocab = data.get_word_dicts()

print('Initialize training set...')
sorted_questions, sorted_answers = data.get_sorted_qa()

print('Initialize graph...')
train_graph = tf.Graph()

with train_graph.as_default():

    print('Initialize model...')
    # Create placeholders for inputs to the model, which are initially empty
    input_data, targets, learn_rate, keep_prob = model_inputs()

    # Sequence length will be the max line length for each batch
    sequence_length = \
        tf.placeholder_with_default(max_length,
                                    None,
                                    name='sequence_length')

    # Find the shape of the input data for sequence_loss
    input_shape = tf.shape(input_data)

    # Create training and inference logits
    train_logits, inference_logits = \
        seq2seq_model(tf.reverse(input_data, [-1]),
                      targets,
                      keep_prob,
                      batch_size,
                      sequence_length,
                      len(vocab_to_int),
                      enc_embed_size,
                      dec_embed_size,
                      rnn_size,
                      num_layers,
                      vocab_to_int,
                      attn_length)

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
            tf.train.AdamOptimizer(lr)

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
    ((len(train_questions)) // batch_size // 2) - 1


print('Training preparation finished @ {}\n'.format(str(datetime.now())))


# ---------- Training ----------
# Add Padding to each sentence in the batch (sorting sentences beforehand -> bucketing)
def pad_sentence_batch(sentence_batch, pad_id):

    max_sentence = \
        max([len(sentence) for sentence in sentence_batch])

    return [sentence + [pad_id] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


# Batch questions and answers together
def batch_data(questions, answers, b_size, pad_id):

    for b_i in range(0, len(questions) // b_size):

        start_i = b_i * b_size

        q_b = questions[start_i:start_i + b_size]
        a_b = answers[start_i:start_i + b_size]

        pad_q_b = np.array(pad_sentence_batch(q_b, pad_id))
        pad_a_b = np.array(pad_sentence_batch(a_b, pad_id))

        yield (pad_q_b, pad_a_b)


print('\nTraining started @ {}'.format(str(datetime.now())))

print('Start session...')
with tf.Session(graph=train_graph) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Check if checkpoint exists...')
    if tf.train.checkpoint_exists(checkpoint):
        print('Load checkpoint...')
        saver.restore(sess, checkpoint)

    for epoch_i in range(1, epochs + 1):

        for batch_i, (questions_batch_i, answers_batch_i) \
                in enumerate(batch_data(train_questions,
                                        train_answers,
                                        batch_size,
                                        vocab_to_int['<PAD>'])):

            start_time = time.time()

            _, loss = \
                sess.run([train_op, cost],
                         {input_data: questions_batch_i,
                          targets: answers_batch_i,
                          learn_rate: lr,
                          sequence_length: answers_batch_i.shape[1],
                          keep_prob: keep_probability})

            total_train_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0:
                print('Epoch {}/{} -+- Batch {}/{} -+- Loss: {} -+- Seconds: {}'.format(
                    epoch_i,
                    epochs,
                    batch_i,
                    len(train_questions) // batch_size,
                    round(total_train_loss / display_step, 4),
                    round(batch_time * display_step)
                ))
                total_train_loss = 0

            if batch_i % validation_check == 0 and batch_i > 0:
                total_valid_loss = 0
                start_time = time.time()

                for batch_ii, (questions_batch_ii, answers_batch_ii) \
                        in enumerate(batch_data(valid_questions,
                                                valid_answers,
                                                batch_size,
                                                vocab_to_int['<PAD>'])):

                    valid_loss = \
                        sess.run(cost,
                                 {input_data: questions_batch_ii,
                                  targets: answers_batch_ii,
                                  learn_rate: lr,
                                  sequence_length: answers_batch_ii.shape[1],
                                  keep_prob: 1})

                    total_valid_loss += valid_loss

                end_time = time.time()
                batch_time = end_time - start_time
                avg_valid_loss = round(total_valid_loss / (len(valid_questions) / batch_size), 4)

                print('Valid Loss: {} ----- Seconds: {} ----- Time: {}'.format(avg_valid_loss,
                                                                               round(batch_time),
                                                                               str(datetime.now())))

                # Reduce learning rate, but not below its minimum value
                lr *= lr_decay
                if lr < min_lr:
                    lr = min_lr

                summary_valid_loss.append(avg_valid_loss)
                if avg_valid_loss <= min(summary_valid_loss):
                    print('New Record!')
                    stop_early = 0
                    saver.save(sess, checkpoint)

                else:
                    print('No Improvement.')
                    stop_early += 1
                    if stop_early == stop:
                        break

        if stop_early == stop:
            print('Stopping Training.')
            break


print('Training finished @ {}\n'.format(str(datetime.now())))


# ---------- Testing ----------
# Choose random input question
r = np.random.choice(len(sorted_questions))
input_question = sorted_questions[r]

# Prepare question
input_question = \
    data.q_to_seq(input_question, vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load the saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    # Load the tensors to be used as inputs
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    answer_logits = sess.run(logits,
                             {input_data: [input_question],
                              keep_prob: 1})[0]

    print('Question')
    print('  Word Ids:       {}'.format([i for i in input_question]))
    print('  Input Words:    {}'.format([int_to_vocab[i] for i in input_question]))

    print('\nAnswer')
    print('  Word Ids:       {}'.format([i for i in np.argmax(answer_logits, 1)]))
    print('  Response Words: {}'.format([int_to_vocab[i] for i in np.argmax(answer_logits, 1)]))
