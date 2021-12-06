import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - max(x)
    new_x = np.exp(x) / np.sum(np.exp(x))
    return new_x


def fprop(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    #z1 = z1 / 256
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    #z2 = z2 / 256
    sm = softmax(z2)
    # loss = -np.sum(yth.dot(np.log(sm)))
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'sm': sm}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    x, y, z1, h1, z2, sm = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'sm')]

    yth = np.zeros((np.size(sm)))
    yth[y] = 1
    yth = yth.reshape(np.size(sm), 1)

    dz2 = (sm - yth)                                                  # dL/dz2
    dW2 = np.dot(dz2, h1.T)                                         # dL/dz2 * dz2/dw2
    db2 = dz2                                                       # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 (sm - yth)) * sigmoid(z1) * (1 - sigmoid(z1))        # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)                                          # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                                                       # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def predict(params, x):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    #z1 = z1 / 256
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    #z2 = z2 / 256
    sm = softmax(z2)
    index_max = np.argmax(sm)
    return index_max


start = timeit.default_timer()

# getting files
train_x_fname, train_y_fname, test_x_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x = np.loadtxt(train_x_fname)
train_y = np.loadtxt(train_y_fname, dtype=int)
test_x = np.loadtxt(test_x_fname)

stop = timeit.default_timer()
print('Time: ', stop - start)

#print(train_x[0])
#print(np.size(train_x, 1))
#print(np.unique(train_y))

INPUT_SIZE = 784
OUTPUT_SIZE = 10
FIRST_HIDDEN_LAYER = 150
NUM_OF_TRAIN = np.size(train_x, 0)
NUM_OF_TEST = np.size(test_x, 0)

# normalization of train_x and test_x
train_x = train_x / 256
test_x = test_x / 256


# creating w, b
"""mu, sigma = 0, 0.05
W1 = np.random.normal(mu, sigma, size=(FIRST_HIDDEN_LAYER, INPUT_SIZE))
b1 = np.random.normal(mu, sigma, size=(FIRST_HIDDEN_LAYER, 1))
W2 = np.random.normal(mu, sigma, size=(OUTPUT_SIZE, FIRST_HIDDEN_LAYER))
b2 = np.random.normal(mu, sigma, size=(OUTPUT_SIZE, 1))
params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}"""


# creating w, b
W1 = np.random.rand(FIRST_HIDDEN_LAYER, INPUT_SIZE)
b1 = np.random.rand(FIRST_HIDDEN_LAYER, 1)
W2 = np.random.rand(OUTPUT_SIZE, FIRST_HIDDEN_LAYER)
b2 = np.random.rand(OUTPUT_SIZE, 1)
params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

W1 = W1 / np.sqrt(1/INPUT_SIZE)
b1 = b1 / np.sqrt(1/INPUT_SIZE)
W2 = W2 / np.sqrt(1/INPUT_SIZE)
b2 = b2 / np.sqrt(1/INPUT_SIZE)


start = timeit.default_timer()
# learning
eta = 0.1
for j in range(50):
    p = np.random.permutation(len(train_x))
    train_x_sh, train_y_sh = train_x[p], train_y[p]
    for i in range(NUM_OF_TRAIN):
        fprop_cache = fprop(train_x_sh[i].reshape(INPUT_SIZE, 1), train_y_sh[i], params)
        bprop_cache = bprop(fprop_cache)

        db1, dW1, db2, dW2 = [bprop_cache[key] for key in ('b1', 'W1', 'b2', 'W2')]
        W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
        params['W1'] = W1 - eta * dW1
        params['b1'] = b1 - eta * db1
        params['W2'] = W2 - eta * dW2
        params['b2'] = b2 - eta * db2

print("done learning")
stop = timeit.default_timer()
print('Time learning: ', stop - start)



start = timeit.default_timer()
# prediction
yth_predict = np.empty((NUM_OF_TEST))
for i in range(NUM_OF_TEST):
    yth_predict[i] = predict(params, test_x[i].reshape(INPUT_SIZE, 1))

stop = timeit.default_timer()
print('Time predict: ', stop - start)

np.savetxt(out_fname + ".txt", yth_predict, fmt='%i')
#print(yth_predict)
"""plt.imshow(train_x[0].reshape(28,-1), cmap='gray')
plt.show()
plt.imshow(train_x[1001].reshape(28,-1), cmap='gray')
plt.show()
plt.imshow(train_x[3578].reshape(28,-1), cmap='gray')
plt.show()"""