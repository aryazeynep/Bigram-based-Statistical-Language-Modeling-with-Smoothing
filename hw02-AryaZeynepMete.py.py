
# -*- coding: utf-8 -*-

"""
Created on Sat Mar  8 15:06:51 2025

@author: ilyas
"""

# Arya Zeynep Mete
# 2210356104

import math
import random
import re
import codecs


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""
    
    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0 # total number of tokens in the train file
        self.sizeOfVocab = 0 # number of unique tokens in the training data
        self.numOfSentences = 0 # total number of sentences in the train file
        self.sentences = [] # list of tokenized (and lowercased) sentences from the training file.
                            # each tokenized sentence should be represented as a list of tokens
                            # the first token of a tokenized sentence should be the special sentence-start token "<s>", 
                            # and the last token should be the special sentence-end token "</s>".
        # TO DO 
        self.vocabulary = [] 
        self.unigrams = []
        self.bigramsList = []

        # instance variable(s) to store the necessary information about ngrams (bigrams and unigrams) from the training file


    # INSTANCE METHODS
    def trainFromFile(self,fn):
        # TO DO
        # 1-) read each line of the file sequentially (one by one)
        f = open(fn, "r")
        
        # 2-) each line should be tokenized using the specific regular expression
        tokens2d = []
        for x in f:
            # All words in a tokenized sentence should be lowercased. Before converting other characters to 
            # lowercase, explicitly replace the Turkish uppercase letters "I" and "İ" with "ı" and "i", respectively.
            x = x.replace("I", "ı").replace("İ", "i").lower()
            
            tokens = re.findall(r"""(?x)  
                                (?:[A-ZÇĞIİÖŞÜ]\.)+              
                                | \d+(?:\.\d*)?(?:\'\w+)?   
                                | \w+(?:-\w+)*(?:\'\w+)?  
                                | \.\.\.  
                                | [][,;.?():_!#^+$%&><|/{()=}\"\'\\\"\`-] 
                                """ , x)
            tokens2d.append(tokens)
        
        # 3-) For each tokenized line, a list of tokenized sentences should be created. 
            # Assume that each sentence ends with one of the following end-of-sentence tokens: ".", "?", or "!". 
            # The last sentence of a line may not necessarily end with an end-of-sentence token. 
            # Any lines that contain no tokens should be skipped. 
            # Each sentence should be padded with a special sentence-start token "<s>" and a special sentence-end token "</s>". 
            # This means that the first token of each tokenized sentence should be "<s>", and the last token should be "</s>". 
        
        newTokenizedSentence = []
        finalTokenizedSentences = []
        for sentences in tokens2d:
            tokensMini = []
            for i in sentences:
                if ((i == ('.')) | (i == ("?")) | (i == ("!"))):
                    tokensMini.append(i)
                    newTokenizedSentence.append("<s>")
                    newTokenizedSentence.extend(tokensMini)
                    newTokenizedSentence.append("</s>")
                    if i not in self.vocabulary: self.vocabulary.append(i)
                    if "<s>" not in self.vocabulary: self.vocabulary.append("<s>")
                    if "</s>" not in self.vocabulary: self.vocabulary.append("</s>")
                    self.numOfTokens = self.numOfTokens +3
                    tokensMini = []
                    finalTokenizedSentences.append(newTokenizedSentence)
                    newTokenizedSentence = []
                elif ((i != (".")) | (i != ("?")) | (i != ("!"))):
                    tokensMini.append(i)
                    if i not in self.vocabulary: self.vocabulary.append(i)
                    self.numOfTokens = self.numOfTokens +1
            if tokensMini != []: #end of sentence with no end line characters
                newTokenizedSentence.append("<s>")
                newTokenizedSentence.extend(tokensMini)
                newTokenizedSentence.append("</s>")
                if "<s>" not in self.vocabulary: self.vocabulary.append("<s>")
                if "</s>" not in self.vocabulary: self.vocabulary.append("</s>")
                self.numOfTokens = self.numOfTokens +3
                finalTokenizedSentences.append(newTokenizedSentence)
                newTokenizedSentence = []

        self.sentences = finalTokenizedSentences
        self.numOfSentences = len(self.sentences)
        self.sizeOfVocab = len(self.vocabulary)


    
    def vocab(self):
        # TO DO
        # The instance method vocab() should return the vocabulary list (unigrams). 
        # Each item in this list must be a tuple (word, frequency), where frequency is the count of occurrences of the word. 
        # The list must be sorted first in descending order by frequency, and then in ascending order by word.

        # LM Sorted Vocabulary (Unigrams) with Frequencies:   
        # [('.', 7), ('</s>', 7), ('<s>', 7), ('c', 7), ('b', 6), ('d', 5), ('a', 4), ('e', 3), ('f', 2)] 
    
        unorderedUnigrams = []
        for v in self.vocabulary:
            vCount = 0
            for s in self.sentences:
                for t in s:
                    if v == t:
                        vCount = vCount + 1
            uniTuple = (v, vCount)
            unorderedUnigrams.append(uniTuple)

        # Sort the list first by frequency in descending order, then by word in ascending order
        self.unigrams = sorted(unorderedUnigrams, key=lambda x: (-x[1], x[0]))
        return self.unigrams
        
  
    def bigrams(self):
        # TO DO
        # LM Sorted Bigrams with Frequencies:   
        # [(('.', '</s>'), 7), (('c', 'd'), 5), (('d', '.'), 5),(('<s>', 'a'), 4), (('b', 'c'), 4), (('<s>', 'e'), 3),  
        # (('a', 'b'), 3), (('b', 'b'), 2), (('c', 'f'), 2), (('e', 'c'), 2),(('f', '.'), 2), (('a', 'c'), 1), (('e', 'b'), 1)]

        # The instance method bigrams() should return a list of bigrams. Each item in the list must be a tuple 
        # ((word1, word2),frequency), where frequency is the count of occurrences of the bigram (word1,word2). 
        # The list must be sorted first in descending order by frequency, and then in ascending order by bigram. 

        unorderedBigrams = []
        for v1 in self.vocabulary:
            for v2 in self.vocabulary:
                vCount = 0
                for s in self.sentences:
                    for t in range(len(s)-1):
                        if ((v1 == s[t]) & (v2 == s[t+1])):
                            vCount = vCount + 1
                if (vCount != 0):
                    biTuple = ((v1, v2), vCount)
                    unorderedBigrams.append(biTuple)

        self.bigramsList = sorted(unorderedBigrams, key=lambda x: (-x[1], x[0]))
        return self.bigramsList


    def unigramCount(self, word):
        # TO DO
        # This method should return the frequency of the unigram word. 
        isOk = False
        for i in self.unigrams:
            if (i[0] == word):
                isOk = True
                return i[1]
        if (isOk == False):
            return 0

        

    def bigramCount(self, bigram):
        # TO DO
        # This method should return the frequency of bigram which is a bigram (word1,word2). 

        isOk = False
        for i in self.bigramsList:
            if (i[0] == bigram):
                isOk = True
                return i[1]
        if (isOk == False):
            return 0


    def unigramProb(self, word):
        # TO DO
        # returns unsmoothed unigram probability value
        #  This method should return the unsmoothed probability of the unigram word. 
        isOk = False
        for i in self.unigrams:
            if (i[0] == word):
                isOk = True
                return (i[1] / self.numOfTokens)
        if (isOk == False):
            return 0

    def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value
        # This method should return the unsmoothed probability of bigram which is a bigram (word1,word2), 
        # i.e., P(word2|word1) will be returned by this method. 

        isOk = False
        for i in self.bigramsList:
            if (i[0] == bigram):
                isOk = True
                for j in self.unigrams:
                    if (j[0] == bigram[0]):
                        numW1 = j[1]
                return (int(i[1]) / int(numW1))
        if (isOk == False):
            return 0

    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        # returns smoothed unigram probability value

        # This method should return the smoothed probability of the unigram word using add-1 smoothing, 
        # with unknown words also handled via add-1 smoothing. Remember, the smoothed probability of a unigram 
        # can be computed as follows: 
        # P(word) = (freq(word)+1)/(numOfTokens+(sizeOfVoc+1)) 

        # Since sizeOfVocab does not include unknown tokens, we add 1 to the size of the vocabulary 
        # in the denominator to account for unknown tokens. 

        # The smoothed bigram probability for any unknown word must be equal to 
        # 1/(numOfTokens+(sizeOfVocab+1)) since its frequency is 0. 

        isOk = False
        for i in self.unigrams:
            if (i[0] == word):
                isOk = True
                return ((i[1]+1) / (self.numOfTokens + len(self.vocabulary) + 1))
        if (isOk == False):
            return ((1) / (self.numOfTokens + len(self.vocabulary) + 1))



    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value

        # This method should return the smoothed probability of the bigram 
        # (where bigram is a bigram (w1,w2)) using add-1 smoothing, with unknown words also handled by add-1 smoothing. 
        # Remember, the smoothed probability of a bigram can be computed as follows: 
        # P((w1,w2)) = (freq((w1,w2))+1)/(freq(w1)+(sizeOfVoc+1)) 

        # Since sizeOfVocab does not include unknown tokens, we add 1 to the size of the vocabulary 
        # in the denominator to account for unknown tokens. The smoothed bigram probability of 
        # a bigram where w2 is an unknown word and w1 is not an unknown word must be equal to 
        # 1/(freq(w1)+(sizeOfVocab+1)), 
        # while the smoothed bigram probability of a bigram where w1 is an unknown word must be equal to 
        # 1/(sizeOfVocab+1).

        isOk = False
        
        for i in self.bigramsList: # i = random bigram(w1, w2)
            if (i[0] == bigram): # random bigram = we want bigram (both w1 and w2 is valid)
                isOk = True
                for j in self.unigrams: # j = random unigram
                    if (j[0] == bigram[0]): # random unigram = w1
                        numW1 = j[1] # w1 is known
                        return ((int(i[1]) + 1) / (int(numW1) + len(self.vocabulary) + 1))
                        
        if (isOk == False):
            # in the case that not w1 and w2 both valid
            # is w1 known
            for m in self.unigrams: # m = random unigram
                    if (m[0] == bigram[0]): # random unigram = w1
                        numW1 = m[1] 
                        return ((1) / (int(numW1) + len(self.vocabulary) + 1))
            # w1 is unknown
            return ((1) / (len(self.vocabulary) + 1))



    def sentenceProb(self,sent):
        # TO DO 
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values
        
        # This method should return the probability of the given sentence sent using smoothed bigram probability values 
        # (the probability of each bigram is obtained using the bigramProb_SmoothingUNK method). 
        # The given sentence sent should be provided as a list of tokens. 
        # Initialize the probability to 1

        probability = 1.0

        #if unigram
        if (len(sent) == 1):
            probability = self.unigramProb_SmoothingUNK(sent[0])
            return probability

        # iterate through the sentence to calculate the probability of each bigram
        for i in range(len(sent) - 1):
            # current bigram
            bigram = (sent[i], sent[i+1])
            
            # calculate the smoothed bigram probability
            bigram_prob = self.bigramProb_SmoothingUNK(bigram)
            
            # multiply the current probability by the bigram probability
            probability *= bigram_prob
        
        return probability


    def generateSentence(self,sent=["<s>"],maxFollowWords=1,maxWordsInSent=20):
        # TO DO 
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)
        
        # This method generates a random sentence by iteratively selecting the next word using bigrams and a probability-based 
        # top-k sampling approach. It starts with the last word of the sentence sent (an optional argument, 
        # defaulting to ["<s>"]) and randomly selects the next word from the top k (= maxFollowWords, an optional argument 
        # with a default value of 1) words that can follow the last word. 
        # The top k words that can follow a given last word w are those that appear in the top k bigrams (w,followword) 
        # with the highest frequencies. 

        # To select the top k follow-up words for a given word w, first, its bigrams must be sorted in descending order 
        # based on their frequencies, and then the bigrams must be sorted in ascending order. 

        # For example, assume the last word is w, and the top 3 following words (when maxFollowWords=3) are 
        # f1, f2, and f3, with bigram frequencies freq((w,f1))=4, freq((w,f2))=3, and freq((w,f3))=1. 
        # In this case, f1 should be selected with a probability of 4/(4+3+1), 
        # f2 with a probability of 3/(4+3+1), and f3 with a probability of 1/(4+3+1). 
        # This process is repeated for each subsequent word. The selection of a probable follow-up word 
        # for this example can be performed in Python as follows. 

        # x = random.randint(1,8)  
        # # 4+3+1=8, randint produces an integer between 1 and 8 
        # # if 1<=x<=4 select f1 else if 5<=x<=7 select f2 else select f3 
        # Generation stops when the special end-of-sentence token "</s>" is generated or when the number of 
        # generated tokens reaches the maximum sentence length (maxWordsInSent), which is defined by the optional argument, 
        # defaulting to 20. 

        # When the method generateSentence() is invoked without any argument, it should produce the most probable sentence. 

        # first w1 is sent
        generatedSentence = []
        generatedSentence.append(sent[0])
        w1 = sent[0]
        for z in range(maxWordsInSent):
            # select most possible n bigram
            w1w2bigrams = []
            for i in self.bigramsList:
                if i[0][0] == w1:
                    w1w2bigrams.append(i)
            
            
            w1w2bigramsFinal = []
            if (maxFollowWords < len(w1w2bigrams)):
                for j in range (maxFollowWords):
                    w1w2bigramsFinal.append(w1w2bigrams[j])
            else:
                w1w2bigramsFinal = w1w2bigrams
            
            # randint possibility
            sumOfFreq = 0
            for k in w1w2bigramsFinal:
                sumOfFreq = sumOfFreq + int(k[1])
            
            x = random.randint(1,sumOfFreq) 
            # for example if 1<=x<=4 select f1 else if 5<=x<=7 select f2 else select f3 
            newWord = ""
            count = 0
            for m in w1w2bigramsFinal:
                count = count + m[1]
                if x <= count:
                    newWord = m[0][1]
                    break

            # add new word to list, w1 = new word
            generatedSentence.append(newWord)
            w1 = newWord
            if newWord == "</s>":
                break
            
        if generatedSentence[-1] != "</s>":
            generatedSentence.append('</s>')
        
        return generatedSentence
        


lm = ngramLM()
lm.trainFromFile('hw02_tinyTestCorpus.txt')
#lm.trainFromFile('hw02_tinyCorpus.txt')

print("LM numOfTokens: ",lm.numOfTokens) 
print("LM sizeOfVocab: ",lm.sizeOfVocab)
print("LM numOfSentences: ",lm.numOfSentences)
print("LM Sentences: \n",lm.sentences)
print("LM Sorted Vocabulary (Unigrams) with Frequencies: \n",lm.vocab())
print("LM Sorted Bigrams with Frequencies: \n",lm.bigrams())

print(lm.unigramCount('a'))
print(lm.unigramCount('b'))
print(lm.unigramCount('g'))

print(lm.unigramProb('a'))
print(lm.unigramProb('b'))
print(lm.unigramProb('g'))

print(lm.bigramCount(('a','b')))   
print(lm.bigramCount(('b','a')))   
print(lm.bigramCount(('a','g')))   
print(lm.bigramCount(('g','a')))   
print(lm.bigramCount(('g','g')))   

print(lm.bigramProb(('a','b')))   
print(lm.bigramProb(('b','a')))   
print(lm.bigramProb(('g','a')))   
print(lm.bigramProb(('a','g')))   
print(lm.bigramProb(('g','g'))) 

# Smoothed probabilities
print(lm.unigramProb_SmoothingUNK('a')) 
print(lm.unigramProb_SmoothingUNK('b')) 
print(lm.unigramProb_SmoothingUNK('g'))

print(lm.bigramProb_SmoothingUNK(('a','b')))   
print(lm.bigramProb_SmoothingUNK(('b','a')))   
print(lm.bigramProb_SmoothingUNK(('g','a')))   
print(lm.bigramProb_SmoothingUNK(('a','g')))   
print(lm.bigramProb_SmoothingUNK(('g','g')))    
        

# Sentence probabilities 
print(lm.sentenceProb(['<s>','a','f','d','.','</s>']))
#0.00032954358213873794 
print(lm.sentenceProb(['<s>','a','c','d','.','</s>']))  
#0.0027914279898810746 
print(lm.sentenceProb(['<s>','a','b','c','d','.','</s>']))  
#0.0017446424936756713 
print(lm.sentenceProb(['<s>','</s>']))  
#0.0588235294117647 
print(lm.sentenceProb(['<s>']))  
#0.13793103448275862 
print(lm.sentenceProb(['a']))  
#0.08620689655172414 


# Generating Sentences  
print(lm.generateSentence())   
# Most Probable Sentence 
#['<s>', 'a', 'b', 'c', 'd', '.', '</s>'] 
# Randomly generated sentences (different sentences can be generated) 
#print(lm.generateSentence(["<s>"],2,20))
# ['<s>', 'a', 'b', 'c', 'd', '.', '</s>'] 
#print(lm.generateSentence(["<s>"],2,20)) 
# ['<s>', 'a', 'b', 'c', 'f', '.', '</s>'] 
print(lm.generateSentence(["<s>"],3,20)) 
# ['<s>', 'e', 'b', 'c', 'd', '.', '</s>'] 
print(lm.generateSentence(["<s>"],3,20)) 
# ['<s>', 'a', 'b', 'c', 'd', '.', '</s>'] 
#print(lm.generateSentence(["<s>"],2,2)) 
# ['<s>', 'a', 'b', '</s>'] 
#print(lm.generateSentence(["<s>"],2,2)) 
# ['<s>', 'a', 'c', '</s>'] 
#print(lm.generateSentence(["<s>"],2,2)) 
# ['<s>', 'e', 'c', '</s>'] 
#print(lm.generateSentence(["<s>"],2,1)) 
# ['<s>', 'e', '</s>'] 
#print(lm.generateSentence(["<s>"],2,1)) 
# ['<s>', 'a', '</s>'] 
#print(lm.generateSentence(["<s>"],2,0)) 
# ['<s>', '</s>'] 
