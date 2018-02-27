import numpy as np
import tensorflow as tf
import re
import time
from datetime import datetime
from model import seq2seq_model

# Customizable parameters
params = {'max_line_length': 20}


# Load data
def get_data(filename):
    return open(filename,
                encoding='utf-8',
                errors='ignore').read().split('\n')


# Fetch max_line_length
def get_max_line_length():
    return params['max_line_length']


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


lines = \
    get_data('movie_lines.txt')

conv_lines = \
    get_data('movie_conversations.txt')

# Create a dictionary to map each line's id with its text
line_dict = {}
for l in lines:
    line = l.split(' +++$+++ ')
    if len(line) == 5:
        line_dict[line[0]] = line[4]

# Create a list of all of the conversations' lines' ids
convs = \
    [id_list.split(',') for id_list in [l.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(' ','') for l in conv_lines]]

# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(line_dict[conv[i]])
        answers.append(line_dict[conv[i+1]])

clean_questions = \
    [clean_text(q) for q in questions]

clean_answers = \
    [clean_text(a) for a in answers]

# Remove too short and too long questions and answers
min_line_length = 2
max_line_length = 20

# Filter out questions with inappropriate lengths
short_questions_temp = []
short_answers_temp = []

i = 0
for q in clean_questions:
    if len(q.split()) >= min_line_length and len(q.split()) <= max_line_length:
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
questions_int_to_vocab = \
    {v_i: v for v, v_i in questions_vocab_to_int.items()}

answers_int_to_vocab = \
    {v_i: v for v, v_i in answers_vocab_to_int.items()}

# Add the EOS element to every answer
short_answers = \
    [a + ' <EOS>' for a in short_answers]

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

# Sort questions and answers by length of questions
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
