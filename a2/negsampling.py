
def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models
    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.
    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.
    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """
    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE
    #loss = -np.log(sigmoid(np.matmul(outsideVectors[indices[0]], centerWordVec))) -\
    #   sum(np.log(sigmoid(-np.matmul(outsideVectors[indices[1:]], centerWordVec))))
    DDV = np.matmul(outsideVectors[indices], centerWordVec)
    sigDDV = np.zeros(DDV.shape[0])
    sigDDV[0] = sigmoid(DDV[0])
    #print(DDV.shape[0])
    for it in range(1,DDV.shape[0]):
        sigDDV[it] = sigmoid(-DDV[it])
    loss = -sum(np.log(sigDDV))
    #print(sigDDV)
    #print(outsideVectors[indices])
    gradCenterVec = (sigDDV[0]-1)*outsideVectors[indices[0]] +\
        np.matmul(1-sigDDV[1:], outsideVectors[indices[1:]])
    gradOutsideVecs = np.zeros_like(outsideVectors)
    '''
    print(centerWordVec)
    print(sigDDV)
    print(type(sigDDV))
    print(DDV)
    print(type(DDV))
    print(type(centerWordVec))
    print(indices)
    '''
    for idx, index in enumerate(indices):
        gradOutsideVecs[index] += ((sigDDV[idx]-1)*centerWordVec if idx == 0 else (1-sigDDV[idx])*centerWordVec)
    ### Please use your implementation of sigmoid in here.
    ### END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs
