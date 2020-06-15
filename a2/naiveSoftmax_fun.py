
def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models
    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.
    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.
    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """
    ### YOUR CODE HERE
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.
    UTV = np.matmul(outsideVectors, centerWordVec)
    Pvalue = softmax(UTV)
    loss = -np.log(Pvalue[outsideWordIdx])
    gradCenterVec = -outsideVectors[outsideWordIdx]+np.matmul(Pvalue,outsideVectors)
    gradOutsideVecs = np.outer(Pvalue,centerWordVec)
    gradOutsideVecs[outsideWordIdx] -= centerWordVec
    ### END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs
