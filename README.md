# Tensorflow Chatbot
This chatbot uses TensorFlow and the cornell movie dialogs corpus to learn how to chat.

The project consists of 4 main files:
 - data.py
   - contains customizable paramters
   - contains functions to process text files from the corpus into useable lists of sentences
 - model.py
   - contains functions to create a seq2seq model
 - training.py
   - uses data.py and model.py to prepare for training
   - is used to train a model
 - inference.py
   - is used to chat with the bot

This project requires TensorFlow v. 1.0.1.

