import tensorflow as tf
import numpy as np
import data
import time
from tensorflow.contrib.seq2seq import sequence_loss
from datetime import datetime
from model import seq2seq_model


# ---------- Preparations ----------
print('Training preparation started @ {}'.format(str(datetime.now())))
print('Set hyperparameters...')
hparams = {
    'epochs': 5,
    'batch_size': 128,
    'rnn_size': 256,
    'num_layers': 2,
    'encoding_embedding_size': 256,
    'decoding_embedding_size': 256,
    'learning_rate': 0.005,
    'learning_rate_decay': 0.95,
    'min_learning_rate': 0.0001,
    'keep_probability': 0.8
}


print('Start session...')
# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
sess = tf.Session()


print('Initialize placeholders...')
# Create placeholders for inputs to the model, which are initially empty
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

max_line_length = data.params['max_line_length']

# Sequence length will be the max line length for each batch
sequence_length = \
    tf.placeholder_with_default(max_line_length,
                                None,
                                name='sequence_length')

# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)


print('Initialize vocabulary...')
vocab_to_int, int_to_vocab = data.get_word_dicts()


print('Initialize training set...')
sorted_questions, sorted_answers = data.get_sorted_qa()

print('Initialize model...')
# Create training and inference logits
train_logits, inference_logits = \
    seq2seq_model(tf.reverse(input_data, [-1]),
                  targets,
                  keep_prob,
                  hparams['batch_size'],
                  sequence_length,
                  len(vocab_to_int),
                  hparams['encoding_embedding_size'],
                  hparams['decoding_embedding_size'],
                  hparams['rnn_size'],
                  hparams['num_layers'],
                  vocab_to_int)

# Create a tensor for inference logits, needed for loading checkpoints
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
        tf.train.AdamOptimizer(hparams['learning_rate'])

    # Gradient Clipping
    gradients = \
        optimizer.compute_gradients(cost)

    capped_gradients = \
        [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

    train_op = \
        optimizer.apply_gradients(capped_gradients)


print('Split training set into training and validating data...')
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


print('Initialize training parameters...')
tparams = {
    # Check training loss every x batches
    'display_step': 100,

    # If validation loss does decrease in x consectutive checks, stop training
    'stop': 5,

    # Path to checkpoint file
    'checkpoint': './best_model.ckpt'
}

# Record training loss for each display step
total_train_loss = 0

# Record validation loss for saving improvements in the model
summary_valid_loss = []

# Variable to keep track of consecutive validation losses
stop_early = 0

# Modulus for checking validation loss (check when 50% and 100% are done)
validation_check = \
    ((len(train_questions)) // hparams['batch_size'] // 2) - 1


print('Start session...')
sess.run(tf.global_variables_initializer())


print('Check if Saver exists...')
if tf.train.checkpoint_exists(tparams['checkpoint']):
    print('Load Saver...')
    saver = tf.train.import_meta_graph(tparams['checkpoint'] + '.meta')
    saver.restore(sess, tparams['checkpoint'])
else:
    print('Create Saver...')
    saver = tf.train.Saver()


print('Training preparation finished @ {}\n'.format(str(datetime.now())))


# ---------- Training ----------
# Add Padding to each sentence in the batch (sorting sentences beforehand -> bucketing)
def pad_sentence_batch(sentence_batch, pad_id):

    max_sentence = \
        max([len(sentence) for sentence in sentence_batch])

    return [sentence + [pad_id] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# Batch questions and answers together
def batch_data(questions, answers, batch_size, vti):

    for batch_i in range(0, len(questions)//batch_size):

        start_i = batch_i * batch_size

        questions_batch = answers[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]

        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, vti['<PAD>']))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, vti['<PAD>']))

        yield pad_questions_batch, pad_answers_batch


print('\nTraining started @ {}'.format(str(datetime.now())))
for epoch_i in range(1, hparams['epochs'] + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(batch_data(train_questions, train_answers, hparams['batch_size'], vocab_to_int)):
        start_time = time.time()

        _, loss = \
            sess.run([train_op, cost],
                     {input_data: questions_batch,
                      targets: answers_batch,
                      lr: hparams['learning_rate'],
                      sequence_length: answers_batch.shape[1],
                      keep_prob: hparams['keep_probability']})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % tparams['display_step'] == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'.format(epoch_i,
                                                                                             hparams['epochs'],
                                                                                             batch_i,
                                                                                             len(train_questions) // hparams['batch_size'],
                                                                                             total_train_loss / tparams['display_step'],
                                                                                             batch_time * tparams['display_step']))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()

            for batch_ii, (questions_batch, answers_batch) in enumerate(batch_data(valid_questions, valid_answers, hparams['batch_size'], vocab_to_int)):
                valid_loss = \
                    sess.run(cost,
                             {input_data: questions_batch,
                              targets: answers_batch,
                              lr: hparams['learning_rate'],
                              sequence_length: answers_batch.shape[1],
                              keep_prob: 1})
                total_valid_loss += valid_loss

            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / hparams['batch_size'])

            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}, Time: {}'.format(avg_valid_loss, batch_time, str(datetime.now())))

            # Reduce learning rate, but not below its minimum value
            # not sure about the lr here
            hparams['learning_rate'] *= hparams['learning_rate_decay']
            if hparams['learning_rate'] < hparams['min_learning_rate']:
                hparams['learning_rate'] = hparams['min_learning_rate']

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver.save(sess, tparams['checkpoint'])

            else:
                print('No Improvement.')
                stop_early += 1
                if stop_early == tparams['stop']:
                    break

    if stop_early == tparams['stop']:
        print('Stopping Training.')
        break

# -------------------------------------- TESTING ------------------


# Prepare question for the model
def question_to_seq(question, vocab_to_int):
    question = data.clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


# Create input question
input_question = 'How are you?'

# Use question from data as input
#random = np.random.choice(len(short_questions))
#input_question = short_questions[random]

# Prepare question
input_question = \
    question_to_seq(input_question, vocab_to_int)

# Pad the question until it equals the max line length
input_question = \
    input_question + [vocab_to_int['<PAD>']] * (max_line_length - len(input_question))

# Add empty questions so input data is correct shape
batch_shell = \
    np.zeros((hparams['batch_size'], max_line_length))

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
