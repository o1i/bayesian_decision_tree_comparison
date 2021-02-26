import pickle

from keras.datasets import mnist


with open("data/mnist.pkl", "wb") as f:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape([train_X.shape[0], -1])
    test_X = test_X.reshape([test_X.shape[0], -1])
    out = (train_X, train_y), (test_X, test_y)
    pickle.dump(out, f)
