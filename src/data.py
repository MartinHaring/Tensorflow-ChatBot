import re

params = {'max_line_length': 20,
          'min_line_length': 2,
          'threshold': 10}


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


# Load all lines from a file
def load_lines(filename):
    return open(filename,
                encoding='utf-8',
                errors='ignore').read().split('\n')


# Create dicts to provide unique ints for each word and vice versa
def get_all_vocabs():

    questions_vocab_to_int, answers_vocab_to_int = \
        get_questions_answers_vocab_to_int()

    questions_int_to_vocab = \
        {v_i: v for v, v_i in questions_vocab_to_int.items()}

    answers_int_to_vocab = \
        {v_i: v for v, v_i in answers_vocab_to_int.items()}

    return questions_vocab_to_int, answers_vocab_to_int, questions_int_to_vocab, answers_int_to_vocab


# Create a dictionary for the frequency of the vocabulary
def get_vocab():

    short_questions, short_answers = \
        get_short_questions_answers()

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

    return vocab


# Create dicts to provide unique ints for each word
def get_questions_answers_vocab_to_int():

    vocab = get_vocab()

    questions_vocab_to_int = {}
    answers_vocab_to_int = {}

    threshold = params['threshold']

    word_id = 0
    for word, frequency in vocab.items():
        if frequency >= threshold:
            questions_vocab_to_int[word] = word_id
            answers_vocab_to_int[word] = word_id
            word_id += 1


    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

    questions_vocab_to_int = \
        add_codes(codes, questions_vocab_to_int)

    answers_vocab_to_int = \
        add_codes(codes, answers_vocab_to_int)

    return questions_vocab_to_int, answers_vocab_to_int


# Add unique elements to vocabs
def add_codes(codes, vocab_to_int):

    for c in codes:
        vocab_to_int[c] = len(vocab_to_int) + 1

    return vocab_to_int


# Create a list of all of the conversations' lines' ids
def get_convs():

    conv_lines = load_lines('movie_conversations.txt')

    convs = \
        [id_list.split(',') for id_list
         in [l.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(' ', '') for l in conv_lines]]

    return convs


# Create a dictionary to map each line's id with its text
def get_line_dict():

    lines = load_lines('movie_lines.txt')

    line_dict = {}
    for l in lines:
        line = l.split(' +++$+++ ')
        if len(line) == 5:
            line_dict[line[0]] = line[4]

    return line_dict


# Sort the sentences into questions (inputs) and answers (targets)
def get_questions_answers():

    convs = get_convs()
    line_dict = get_line_dict()

    questions = []
    answers = []

    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(line_dict[conv[i]])
            answers.append(line_dict[conv[i+1]])

    return questions, answers


def get_clean_questions_answers():

    questions, answers = \
        get_questions_answers()

    clean_questions = \
        [clean_text(q) for q in questions]

    clean_answers = \
        [clean_text(a) for a in answers]

    return clean_questions, clean_answers


# Filter out questions with inappropriate lengths. Also, add the EOS element to every answer
def get_short_questions_answers():

    clean_questions, clean_answers = get_clean_questions_answers()

    min_line_length = params['min_line_length']
    max_line_length = params['max_line_length']

    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for q in clean_questions:
        if len(q.split()) >= min_line_length and len(q.split()) <= max_line_length:
            short_questions_temp.append(q)
            short_answers_temp.append(clean_answers[i])
        i += 1

    short_questions = []
    short_answers = []

    i = 0
    for a in short_answers_temp:
        if len(a.split()) >= min_line_length and len(a.split()) <= max_line_length:
            short_answers.append(a)
            short_questions.append(short_questions_temp[i])
        i += 1

    short_answers = \
        [a + ' <EOS>' for a in short_answers]

    return short_questions, short_answers


# Convert the text to ints and replace rare words with <UNK>
def get_int_questions_answers():

    short_questions, short_answers = get_short_questions_answers()
    questions_vocab_to_int, answers_vocab_to_int = get_questions_answers_vocab_to_int()

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

    return int_questions, int_answers


# Fetch sorted_questions & sorted_answers
def get_sorted_questions_answers():

    max_line_length = params['max_line_length']

    int_questions, int_answers = get_int_questions_answers()

    sorted_questions = []
    sorted_answers = []

    for length in range(1, max_line_length+1):
        for i in enumerate(int_questions):
            if len(i[1]) == length:
                sorted_questions.append(int_questions[i[0]])
                sorted_answers.append(int_answers[i[0]])

    return sorted_questions, sorted_answers