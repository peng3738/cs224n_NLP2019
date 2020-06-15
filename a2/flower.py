


word2vecModel = skipgram;
word2Ind = dummy_tokens
wordVectors = dummy_vectors
dataset
windowSize=5
word2vecLossAndGradient = naiveSoftmaxLossAndGradient

    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        print(windowSize1)
        print('\n')
        centerWord, context = dataset.getRandomContext(windowSize1)
        print(centerWord)
        print('\n')
        print(context)
        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

















f = lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient)
x0 =  wordVectors
step = 0.03
iterations = 40000
postprocessing = None
useSaved = True
PRINT_EVERY = 10

