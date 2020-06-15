dataset = type('dummy', (), {})()


def dummySampleTokenIdx():
    return random.randint(0, 4)


def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0, 4)], \
           [tokens[random.randint(0, 4)] for i in range(2 * C)]


dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalizeRows(np.random.randn(10, 3))
dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
