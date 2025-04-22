# Bigram-based-Statistical-Language-Modeling-with-Smoothing
This project is a Python implementation of a statistical Bigram Language Model. The model is designed to process natural language data, compute unigram and bigram frequencies, calculate probabilities, and generate sentences using a probabilistic sampling method.

## Features
Tokenization using regular expressions

Sentence boundary detection and padding with special tokens <s> and </s>

Lowercasing with proper handling of Turkish characters (I → ı, İ → i)

Vocabulary generation with frequency-based sorting

Bigram frequency extraction and sorted output

## Probability computation for:

Unsmoothed unigram and bigram

Add-one (Laplace) smoothed unigram and bigram with unknown word handling

Sentence probability calculation using smoothed bigram model

Random sentence generation using top-k bigram sampling based on frequency-weighted probabilities

## Core Class: ngramLM
Instance Variables:
numOfTokens: Total tokens in training data

sizeOfVocab: Number of unique words

numOfSentences: Number of detected sentences

sentences: Tokenized sentence list, each padded with <s> and </s>

Additional dictionaries to store unigram and bigram counts

Main Methods:
trainFromFile(fn): Trains the model from a UTF-8 encoded text file

vocab(): Returns a sorted list of unigrams with frequencies

bigrams(): Returns a sorted list of bigrams with frequencies

unigramCount(word): Returns frequency of a unigram

bigramCount((w1, w2)): Returns frequency of a bigram

unigramProb(word): Returns unsmoothed unigram probability

bigramProb((w1, w2)): Returns unsmoothed bigram probability

unigramProb_SmoothingUNK(word): Returns smoothed unigram probability (with unknown handling)

bigramProb_SmoothingUNK((w1, w2)): Returns smoothed bigram probability (with unknown handling)

sentenceProb(sent): Computes probability of a sentence (as token list) using smoothed bigram model

generateSentence(sent=["<s>"], maxFollowWords=1, maxWordsInSent=20): Generates a random sentence using top-k bigram sampling

## Notes
Sentence boundaries are detected using ".", "?", and "!"

Tokenization is done with a detailed regular expression to handle abbreviations, numbers, punctuation, etc.

Unknown tokens are handled with add-one smoothing by expanding the vocabulary size accordingly

Sentence generation uses random sampling weighted by bigram frequency to mimic real language flow

