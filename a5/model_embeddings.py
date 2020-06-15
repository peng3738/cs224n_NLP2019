#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.echar = 50
        pad_token_idx = vocab.word2id['<pad>']
        #print("vocab length")
        #print(len(vocab.word2id))
        #print(len(vocab.char2id))
        #self.embeddings = nn.Embedding(len(vocab.word2id), self.echar, padding_idx=pad_token_idx)
        self.embeddings = nn.Embedding(len(vocab.char2id), self.echar, padding_idx=pad_token_idx)
        self.Highwaymodule = Highway(self.embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        #print(input.shape)
        #print(input.type())
        #print(input)
        [sentence_length, batch_size, m_word] = list(input.shape)
        x_emb = self.embeddings(input)
        #print(x_emb.shape)
        x_reshaped = torch.transpose(x_emb, 2, 3)

        x_reshaped_d = torch.reshape(x_reshaped, [sentence_length * batch_size, self.echar, m_word])
        CNNmodule = CNN(self.embed_size, self.echar, 5, m_word)
        x_conv_out = CNNmodule(x_reshaped_d)
        #print(x_conv_out.shape)
        x_highway = self.Highwaymodule(x_conv_out)
        #print(x_highway.shape)
        x_word_emb_d = self.dropout(x_highway)
        #print(x_word_emb_d.shape)
        x_word_emb = torch.reshape(x_word_emb_d, [sentence_length, batch_size, self.embed_size])
        #print(x_word_emb.shape)
        return x_word_emb
        ### YOUR CODE HERE for part 1j
        ### END YOUR CODE

