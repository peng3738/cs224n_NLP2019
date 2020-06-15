def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec
    Implement the skip-gram model in this function.
    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.
    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE
    idxc=word2Ind[currentCenterWord]
    for word in outsideWords:
        idxo = word2Ind[word]
        loss_unit,gradCenterVecs_unit,gradOutsideVectors_unit= \
            word2vecLossAndGradient(centerWordVectors[idxc],idxo,outsideVectors,dataset)
        loss += loss_unit
        gradCenterVecs[word2Ind[currentCenterWord]] += gradCenterVecs_unit
        gradOutsideVectors += gradOutsideVectors_unit
    ### END YOUR CODE
    return loss, gradCenterVecs, gradOutsideVectors
