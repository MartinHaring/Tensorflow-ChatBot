

# ---------- Testing ----------
# Prepare question for the model
def question_to_seq(question, vti):
    question = data.clean_text(question)
    return [vti.get(word, vti['<UNK>']) for word in question.split()]


# Create input question
input_question = 'How are you?'

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
