# NLP-with-Sequence-Models

In Course 3 of the Natural Language Processing Specialization, I :

a) Train a neural network with GLoVe word embeddings to perform sentiment analysis of tweets,
b) Generate synthetic Shakespeare text using a Gated Recurrent Unit (GRU) language model,
c) Train a recurrent neural network to perform named entity recognition (NER) using LSTMs with linear layers, and 
d) Use so-called ‘Siamese’ LSTM models to compare questions in a corpus and identify those that are worded differently but have the same meaning.


## Assignment 1: Sentiment with Deep Neural Networks
The first assignment of course 3. In this assignment, I explored sentiment analysis using deep neural networks.

In course 1, I implemented Logistic regression and Naive Bayes for sentiment analysis. However if I gave my old models an example like: <br>

<p align='center'>
<b>This movie was almost good. </b> <br>
</p>

My model would have predicted a positive sentiment for that review. However, that sentence has a negative sentiment and indicates that the movie was not good. To solve those kinds of misclassifications, I wrote a program that uses deep neural networks to identify sentiment in text. By completing this assignment, I :

<li>Understand how you can build/design a model using layers</li>
<li>Train a model using a training loop</li>
<li>Use a binary cross-entropy loss function</li>
<li>Compute the accuracy of your model</li>
<li>Predict using your own input</li>


## Assignment 2: Deep N-grams
The second assignment of course 3. In this assignment I explored Recurrent Neural Networks RNN.

I used the fundamentals of google's trax package to implement any kind of deeplearning model.
By completing this assignment, I learnt how to implement models from scratch:

<li>How to convert a line of text into a tensor</li>
<li>Create an iterator to feed data to the model</li>
<li>Define a GRU model using trax</li>
<li>Train the model using trax</li>
<li>Compute the accuracy of your model using the perplexity</li>
<li>Predict using your own model</li>
