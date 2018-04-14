import re

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


# Fill question and answer lists with sentences that have appropriate lengths
def fill_short_qa(short_q, short_a, clean_q, clean_a):

    min_length = dparams['min_line_length']
    max_length = dparams['max_line_length']

    i = 0
    for sent in clean_q:
        if len(sent.split()) >= min_length and len(sent.split()) <= max_length:
            short_q.append(sent)
            short_a.append(clean_a[i])
        i += 1

    return short_q, short_a


# Format questions and answers appropriately.
def get_short_qa():

    questions, answers = get_qa()

    clean_questions = [clean_text(q) for q in questions]
    clean_answers = [clean_text(a) for a in answers]

    short_questions_temp, short_answers_temp = \
        fill_short_qa([], [],
                      clean_questions,
                      clean_answers)

    short_answers, short_questions = \
        fill_short_qa([], [],
                      short_answers_temp,
                      short_questions_temp)

    return short_questions, short_answers


# Create a vocabulary, containing the frequency of each word of a given list
def fill_vocab(vocab, short_qa):

    for qa in short_qa:
        for word in qa.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    return vocab


# Fill a dict that maps words with indeces, ignore rare words
def fill_vti(vti, vocab, threshold):

    word_id = 0
    for word, frequency in vocab.items():
        if frequency >= threshold:
            vti[word] = word_id
            word_id += 1

    return vti


# Add unique elements to vocabs
def add_codes(codes, vti):

    for c in codes:
        vti[c] = len(vti) + 1

    return vti


# Create dicts to provide unique indeces for common words; also add unique elements
def get_vocab_to_int():

    short_q, short_a = get_short_qa()
    vocab = fill_vocab({}, short_q + short_a)

    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']
    threshold = dparams['threshold']

    vocab_to_int = fill_vti({}, vocab, threshold)
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
def fill_ints(sent, vti):
    return [vti[word]
            if word in vti
            else vti['<UNK>']
            for word in sent.split()]


# Create lists of sentences, where words are replaced with their indeces
def get_int_qa():

    short_q, short_a = get_short_qa()
    vocab_to_int = get_vocab_to_int()

    short_a = [a + ' <EOS>' for a in short_a]

    int_q = [fill_ints(q, vocab_to_int) for q in short_q]
    int_a = [fill_ints(a, vocab_to_int) for a in short_a]

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
