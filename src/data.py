import numpy as np
import tensorflow as tf
import re
import time

# ---------------------------------------- DATA PREPS -------------------------------

# Load data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8').read().split('\n')

# Create a dictionary to map each line's id with its text
line_dict = {}
for l in lines:
    line = l.split(' +++$+++ ')
    if len(line) == 5:
        line_dict[line[0]] = line[4]

# Create a list of all of the conversations' lines' ids
convs = [id_list.split(',') for id_list in [l.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","") for l in conv_lines]]

# Alternative code
#for l in conv_lines:
#    line = l.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
#    convs.append(line.split(','))

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(line_dict[conv[i]])
        answers.append(line_dict[conv[i+1]])

# function to remove unnecessary characters and to alter word formats
def clean_text(text):
    text = text.lower()

    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"it's", 'it is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"how's", 'how is', text)
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'re", ' are', text)
    text = re.sub(r"\'d", ' would', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"n'", 'ng', text)
    text = re.sub(r"'bout", 'about', text)
    text = re.sub(r"'til", 'until', text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

clean_questions = [clean_text(q) for q in questions]
clean_answers = [clean_text(a) for a in answers]

# Find the length of sentences
#lengths = []
#for q in clean_questions:
#    lengths.append(len(q.split()))
#for a in clean_answers:
#    lengths.append(len(a.split()))

# Create a dataframe so that the values can be inspected
#lengths = pd.DataFrame(lengths, columns=['counts'])
#print(lengths.describe())
#print(np.percentile(lengths, 80))

# Remove too short and too long questions and answers
min_line_length = 2
max_line_length = 20

# Filter out questions with inappropriate lengths
short_questions_temp = []
short_answers_temp = []

i = 0
for q in clean_questions:
    if (len(q.split()) >= min_line_length and len(q.split()) <= max_line_length):
        short_questions_temp.append(q)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out answers with inappropriate lengths
short_questions = []
short_answers = []

i = 0
for a in short_answers_temp:
    if (len(a.split()) >= min_line_length and len(a.split()) <= max_line_length):
        short_answers.append(a)
        short_questions.append(short_questions_temp[i])
    i += 1

# Compare numbers, show usage
#print('# of questions: {}'.format(len(short_questions)))
#print('# of answers: {}'.format(len(short_answers)))
#print('% of data used: {}%'.format(round(len(short_questions)/len(questions),4)*100))

# Create a dictionary for the frequency of the vocabulary
vocab = {}
for q in short_questions:
    for word in q.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for a in short_answers:
    for word in a.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

# Set threshold for rare words
threshold = 10

# Create dicts to provide unique ints for each word.
questions_vocab_to_int = {}
answers_vocab_to_int = {}

word_id = 0
for word,frequency in vocab.items():
    if frequency >= threshold:
        questions_vocab_to_int[word] = word_id
        answers_vocab_to_int[word] = word_id
        word_id += 1

# Add unique elements
codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

for c in codes:
    questions_vocab_to_int[c] = len(questions_vocab_to_int) + 1
    answers_vocab_to_int[c] = len(answers_vocab_to_int) + 1

# Create dictionaries to map the unique integers to words
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

# Check lengths
#print(len(questions_vocab_to_int))
#print(len(questions_int_to_vocab))
#print(len(answers_vocab_to_int))
#print(len(answers_int_to_vocab))

# Add the EOS element to every answer
short_answers = [a + ' <EOS>' for a in short_answers]

# Convert the text to ints and replace rare words with <UNK>
int_questions = []
for q in short_questions:
    ints = []
    for word in q.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    int_questions.append(ints)

int_answers = []
for a in short_answers:
    ints = []
    for word in a.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    int_answers.append(ints)

# Check the lengths
#print(len(questions_int))
#print(len(answers_int))

# Calculate UNK percentage
#word_count = 0
#unk_count = 0

#for q in int_questions:
#    for word in q:
#        if word == questions_vocab_to_int['<UNK>']:
#            unk_count += 1
#        word_count += 1

#for a in int_answers:
#    for word in a:
#        if word == answers_vocab_to_int['<UNK>']:
#            unk_count += 1
#        word_count += 1

#unk_ratio = round(unk_count/word_count,4)*100

#print("Total number of words: {}".format(word_count))
#print("Number of times <UNK> is used: {}".format(unk_count))
#print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))
#print()

# Sort questions and answers by length of questions (for bucketing)
sorted_questions = []
sorted_answers = []

# i is a tuple of index + [int].
# if len([int]) == length,
# the question with the corresponding index is added to sorted.
for length in range(1, max_line_length+1):
    for i in enumerate(int_questions):
        if len(i[1]) == length:
            sorted_questions.append(int_questions[i[0]])
            sorted_answers.append(int_answers[i[0]])

#print(len(sorted_questions))
#print(len(sorted_answers))
#print()
#for i in range(3):
#    print(sorted_questions[i])
#    print(sorted_answers[i])
#    print()

# ---------------------------------------------- MODEL --------------------------

# Create placeholders for inputs to the model
# these are initially empty
def model_inputs():

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

# Remove last word id from each batch and concat <GO> to the start
def process_encoding_input(target_data, vocab_to_int, batch_size):

    ending = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1,1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

# Create the encoding layer
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):

    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs,
                                                   dtype = tf.float32)
    return enc_state

# Decode training data
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob, batch_size):

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                              attention_option="bahdanau",
                                                                                              num_units=dec_cell.output_size)

    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")

    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              dec_embed_input,
                                                              sequence_length,
                                                              scope = decoding_scope)

    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)

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
                         output_fn,
                         keep_prob,
                         batch_size):

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                              attention_option = "bahdanau",
                                                                                              num_units = dec_cell.output_size)

    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         maximum_length,
                                                                         vocab_size,
                                                                         name = "attn_dec_inf")

    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                infer_decoder_fn,
                                                                scope = decoding_scope)

    return infer_logits

# Create decoding cell and input parameters for training and inference decoding layers
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, vocab_to_int, keep_prob, batch_size):

    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                               vocab_size,
                                                               None,
                                                               scope = decoding_scope,
                                                               weights_initializer = weights,
                                                               biases_initializer = biases)

        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)

        decoding_scope.reuse_variables()

        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)

        return train_logits, infer_logits

# Use previous functions to create training and inference logits
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, questions_vocab_to_int):

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       answers_vocab_size + 1,
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))

    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train_logits, infer_logits = decoding_layer(dec_embed_input,
                                                dec_embeddings,
                                                enc_state,
                                                questions_vocab_size,
                                                sequence_length,
                                                rnn_size,
                                                num_layers,
                                                questions_vocab_to_int,
                                                keep_prob,
                                                batch_size)

    return train_logits, infer_logits


# Set Hyperparameters
epochs = 50
batch_size = 64
rnn_size = 256
num_layers = 2
encoding_embedding_size = 256
decoding_embedding_size = 256
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()

# Start session
#sess = tf.InteractiveSession()
sess = tf.Session()

# Load model inputs
input_data, targets, lr, keep_prob = model_inputs()

# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')

# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create training and inference logits
train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                               targets,
                                               keep_prob,
                                               batch_size,
                                               sequence_length,
                                               len(answers_vocab_to_int),
                                               len(questions_vocab_to_int),
                                               encoding_embedding_size,
                                               decoding_embedding_size,
                                               rnn_size,
                                               num_layers,
                                               questions_vocab_to_int)

# Create a tensor for inference logits, needed for loading checkpoints
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):

    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# Padding to each sentence in the batch
def pad_sentence_batch(sentence_batch, vocab_to_int):

    max_sentence = max([len(sentence) for sentence in sentence_batch])

    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

# Batch questions and answers together
def batch_data(questions, answers, batch_size):

    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = answers[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch

# Validate the training with 10% of the data
train_valid_split = int(len(sorted_questions)*0.1)

# Split questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]

#print(len(train_questions))
#print(len(valid_questions))

# ----------------------------------------- Training -------------------

# Check training loss every 100 batches
display_step = 100
stop_early = 0
# If validation loss does decrease in 5 consectutive checks, stop training
stop = 5
# Modulus for checking validation loss
validation_check = ((len(train_questions))//batch_size//2) - 1
# Record training loss for each display step
total_train_loss = 0
# Record validation loss for saving improvements in the model
summary_valid_loss = []

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(batch_data(train_questions, train_answers, batch_size)):
        start_time = time.time()

        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'.format(epoch_i,
                                                                                             epochs,
                                                                                             batch_i,
                                                                                             len(train_questions) // batch_size,
                                                                                             total_train_loss / display_step,
                                                                                             batch_time * display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()

            for batch_ii, (questions_batch, answers_batch) in enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss

            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)

            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_line_length

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break

    if stop_early == stop:
        print("Stopping Training.")
        break

# -------------------------------------- TESTING ------------------


# Prepare question for the model
def question_to_seq(question, vocab_to_int):
    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


# Create input question
input_question = 'Why is the sky blue?'

# Use question from data as input
#random = np.random.choice(len(short_questions))
#input_question = short_questions[random]

# Prepare question
input_question = question_to_seq(input_question, questions_vocab_to_int)

# Pad the question until it equals the max line length
input_question = input_question + [questions_vocab_to_int['<PAD>']] * (max_line_length - len(input_question))

# Add empty questions so input data is correct shape
batch_shell = np.zeros((batch_size, max_line_length))

# Set first question to be out input question
answer_logits = sess.run(inference_logits, {input_data: batch_shell,
                                            keep_prob: 1.0})[0]

# Remove padding from Question and Answer
pad_q = questions_vocab_to_int['<PAD>']
pad_a = answers_vocab_to_int['<PAD>']

print('Question')
print('  Word Ids:       {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words:    {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:       {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
