import re
from collections import Counter

# Data-parameters indicate boundaries for words and lines
dparams = {
    'max_line_length': 18,
    'min_line_length': 2,
    'threshold': 10
}

# Training-parameters provide data used in training
tparams = {
    # Amount of times the training data is processed by the network
    'epochs': 3,

    # Amount of sentences that are processed at once
    'batch_size': 512,

    # Check training loss every x batches
    'display_step': 25,

    # Amount of validation per epoch
    'validations': 4,

    # If validation loss decreases in x consectutive checks, stop training
    'stop': 10,

    # Path to checkpoint file
    'checkpoint': './model-rnn128-2.ckpt'
}

# Hyper-parameters are variables used by the neural net
hparams = {
    'rnn_size': 128,
    'num_layers': 2,
    'encoding_embedding_size': 256,
    'decoding_embedding_size': 256,
    'attn_length': 32,
    'learning_rate': 0.005,
    'learning_rate_decay': 0.95,
    'min_learning_rate': 0.0001,
    'keep_probability': 0.8
}


# Fetch hparams and tparams individually
def fetch_hparams():
    return hparams['rnn_size'], \
           hparams['num_layers'], \
           hparams['encoding_embedding_size'], \
           hparams['decoding_embedding_size'], \
           hparams['attn_length'], \
           hparams['learning_rate'], \
           hparams['learning_rate_decay'], \
           hparams['min_learning_rate'], \
           hparams['keep_probability']


def fetch_tparams():
    return tparams['epochs'], \
           tparams['batch_size'], \
           tparams['display_step'], \
           tparams['validations'], \
           tparams['stop'], \
           tparams['checkpoint']


# Load all lines from a file
def load_lines(filename):
    return open(filename,
                encoding='utf-8',
                errors='ignore').read().split('\n')[:-1]


# Create a dictionary to map each line's id with its text
def create_line_dict(lines):
    return {line[0]: line[4]
            for line
            in [l.split(' +++$+++ ')
                for l in lines]}


# Create a list of all of the conversations' lines' ids
def get_convs(conv_lines, line_dict):
    return [translate_conv(conv_ids, line_dict)
            for conv_ids in [conv.split(',') for conv in extract_convs(conv_lines)]]


def extract_convs(c_lines):
    return [l.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(' ', '') for l in c_lines]


def translate_conv(c, ld):
    return [ld[i] for i in c]


# Sort the sentences into questions (inputs) and answers (targets)
def get_qa():

    line_dict = create_line_dict(load_lines('movie_lines.txt'))
    convs = get_convs(load_lines('movie_conversations.txt'), line_dict)

    questions = [q for c_qs in [conv[:-1] for conv in convs] for q in c_qs]
    answers = [a for c_as in [conv[1:] for conv in convs] for a in c_as]

    return questions, answers


# Remove unnecessary characters and alter word formats
def clean_text(text):

    text = text.lower()
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"it's", 'it is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"there's", 'there is', text)
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


def clean_tuple_map(x):
    return tuple(map(clean_map, x))


def clean_map(x):
    return clean_text(x)


def len_filter(x):
    return len(x[0].split(' ')) >= dparams['min_line_length'] \
           and len(x[0].split(' ')) <= dparams['max_line_length'] \
           and len(x[1].split(' ')) >= dparams['min_line_length'] \
           and len(x[1].split(' ')) <= dparams['max_line_length']


# Format and filter questions and answers appropriately.
def get_filtered_qa():

    questions, answers = get_qa()

    clean_qa = list(map(clean_tuple_map, list(zip(questions, answers))))
    filtered_qa = list(filter(len_filter, clean_qa))
    qa = list(zip(*filtered_qa))

    return qa[0], qa[1]


# Create a vocabulary, containing the frequency of each word of a given list
def create_vocab(short_qa):

    words = [word for sentence
             in [sentence.split() for sentence in short_qa]
             for word in sentence]

    return dict(Counter(words))


# Fill a dict that maps words with indeces, filter out rare words
def get_vti(vocab, threshold):

    filtered_words = list(filter(lambda x: vocab[x] >= threshold, vocab.keys()))

    return {word: filtered_words.index(word) for word in filtered_words}


# Add unique elements to vocabs
def add_codes(codes, vti):

    for c in codes:
        vti[c] = len(vti) + 1

    return vti


# Create dicts to provide unique indeces for common words; also add unique elements
def get_vocab_to_int():

    filtered_q, filtered_a = get_filtered_qa()
    vocab = create_vocab(filtered_q + filtered_a)

    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']
    threshold = dparams['threshold']

    vocab_to_int = get_vti(vocab, threshold)
    vocab_to_int = add_codes(codes, vocab_to_int)

    return vocab_to_int


# Switch places of keys and values of a dict
def reverse_dict(dictionary):
    return {v_i: v for v, v_i in dictionary.items()}


# Create dicts to provide unique indeces for each word and vice versa
def get_word_dicts():

    vocab_to_int = get_vocab_to_int()
    int_to_vocab = reverse_dict(vocab_to_int)

    return vocab_to_int, int_to_vocab


# Convert the text to ints and replace rare words with <UNK>
def convert_to_ints(sent, vti):
    return [vti[word]
            if word in vti
            else vti['<UNK>']
            for word in sent.split()]


# Create lists of sentences, where words are replaced with their indeces
def get_int_qa():

    filtered_q, filtered_a = get_filtered_qa()
    vocab_to_int = get_vocab_to_int()

    filtered_a = [a + ' <EOS>' for a in filtered_a]

    int_q = [convert_to_ints(q, vocab_to_int) for q in filtered_q]
    int_a = [convert_to_ints(a, vocab_to_int) for a in filtered_a]

    return int_q, int_a


# Sort a list of indeces lists on a given basis
def sort_ints(ints, max_length, basis):
    return [ints[i[0]]
            for length in range(1, max_length+1)
            for i in enumerate(basis)
            if len(i[1]) == length]


# Sort questions and answers
def get_sorted_qa():

    max_line_length = dparams['max_line_length']

    int_q, int_a = get_int_qa()

    sorted_q = sort_ints(int_q, max_line_length, int_q)
    sorted_a = sort_ints(int_a, max_line_length, int_q)

    return sorted_q, sorted_a


# Translate a given question into their corresponding indeces indicated by a vocab
def q_to_seq(q, vti):
    q = clean_text(q)
    return [vti.get(word, vti['<UNK>']) for word in q.split()]
