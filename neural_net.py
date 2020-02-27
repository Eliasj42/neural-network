import numpy as np

class Layer:

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, delta, learning_rate):
        raise NotImplementedError

class ReLU(Layer):
    def __init__(self):
        self.a = None
        self.d = None
        return
        

    def forward(self, x):
        self.a = np.maximum(x, 0)
        return self.a

    def backward(self, delta, learning_rate = 0.001):
        self.d = delta * (self.a > 0).astype(int)
        return self.d

class Linear(Layer):

    def __init__(self, n_in, n_out):
        self.n_out = n_out
        self.n_in = n_in
        self.W = np.random.normal(0,1,(n_in, n_out))
        
        self.b = np.zeros((1, n_out))

    def forward(self, x):
        self.x = x
        self.a = np.matmul(x, self.W)
        self.a += self.b
        return self.a

    def backward(self, delta, learning_rate):
        self.d  = np.matmul(delta, np.transpose(self.W))
        self.W = self.W - learning_rate * (np.matmul(np.transpose(self.x), delta))
        self.b = self.b - learning_rate * (delta.sum(axis = 0))

        return self.d

class Loss:

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

class CrossEntropyLoss(Loss):

    def __init__(self):
        pass

    def forward(self, x, y):
        self.sm = np.exp(x)
        self.sm = (self.sm.T / self.sm.sum(axis = 1)).T
        guesses = np.argmax(self.sm, axis = 1)
        acc = (guesses == y).sum()
        cel = -np.log(np.exp(np.choose(y, x.T)) / np.exp(x).sum(axis = 1))
        self.y = y
        return cel.sum(), acc

    def backward(self):
        d = np.zeros(self.sm.shape)
        for i in range(0, d.shape[0]):
          d[i, self.y[i]] = 1
        d -= self.sm
        d = -d
        self.d = d
        return d

class NN:
    def __init__(self, layers, criterion):
        self.layers = layers
        self.criterion = criterion

    def train(self, x, y, learning_rate):
        a = x
        for layer in self.layers:
          a = layer.forward(a)
        loss , accuracy = self.criterion.forward(a, y)
        d = self.criterion.backward()
        for layer in self.layers[::-1]:
          d = layer.backward(d, learning_rate)
          

        return loss, accuracy

    def val(self, x, y):
        a = x
        for layer in self.layers:
          a = layer.forward(a)
        loss , accuracy = self.criterion.forward(a, y)
        return loss, accuracy

if __name__ == '__main__':
	mnist_data = np.load("mnist.npz")
	train_x, train_y = mnist_data["train_x"], mnist_data["train_y"]
	val_x, val_y = mnist_data["val_x"], mnist_data["val_y"]
	train_num, val_num = train_x.shape[0], val_x.shape[0]

	n_epochs = 100
	batch_size = 64
	learning_rate = 1e-3
	layers = [
	    Linear(784, 100),
	    ReLU(),
	    Linear(100, 10)
	]
	criterion = CrossEntropyLoss()
	mnist_nn = NN(layers, criterion)


	# statistic data
	train_loss_list, train_acc_list = [], []
	val_loss_list, val_acc_list = [], []

	# begin training and validation
	for e in range(n_epochs):
	    train_loss, train_acc = 0, 0
	    val_loss, val_acc = 0, 0

	    # shuffle the training set each epoch to prevent overfitting
	    idxs = np.arange(train_num)
	    np.random.shuffle(idxs)
	    train_x, train_y = train_x[idxs], train_y[idxs]

	    # training
	    for b in range(0, train_num, batch_size):
	        range_ = range(b, min(b + batch_size, train_num))
	        loss, accuracy = mnist_nn.train(train_x[range_], train_y[range_], learning_rate)
	        train_loss += loss
	        train_acc += accuracy

	    # validation
	    for b in range(0, val_num, batch_size):
	        range_ = range(b, min(b + batch_size, val_num))
	        loss, accuracy = mnist_nn.val(val_x[range_], val_y[range_])
	        val_loss += loss
	        val_acc += accuracy

	    train_loss /= train_num
	    train_acc /= train_num
	    val_loss /= val_num
	    val_acc /= val_num
	    train_loss_list.append(train_loss)
	    train_acc_list.append(train_acc)
	    val_loss_list.append(val_loss)
	    val_acc_list.append(val_acc)

	    # summary of the epoch
	    print("epoch: {}, train acc: {:.2f}%, train loss: {:.3f}, val acc: {:.2f}%, val loss: {:.3f}"
	          .format(e+1, train_acc*100, train_loss, val_acc*100, val_loss))
