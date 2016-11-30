
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
import numpy as np
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data, 60K trainset and 10K testset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 1. Define Variables and Placeholders
# Model is Y = softmax(X*W + b)

# input f 28 X 28 images with 1 channel (grayscale images) . None is left for minibatch size
X = tf.placeholder(tf.float32,[None, 28,28,1])
# Y_ true labels. One-Hot Encoding
Y_= tf.placeholder(tf.float32,[None,10])
#Weights
W = tf.Variable(tf.zeros([784,10]))
#biases
b = tf.Variable(tf.zeros([10]))

#flatten image
XX = tf.reshape(X,[-1,784])

# 2. Define the model
Y = tf.nn.softmax(tf.matmul(XX,W)+b)

# 3. Define the loss function
cross_entropy =  tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y),reduction_indices=[1]))
#cross_entropy =  -tf.reduce_sum(Y_ * tf.log(Y),reduction_indices=[1])

#prediction
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))

# 4. Define the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# print("entropy "+ str(cross_entropy))
# print("predcition "+str(correct_prediction))
# print("accuracy "+str(accuracy))
# print("train_step_type "+str(type(train_step)))
# print("train_step " +str(train_step))

# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


def training_step(num_iter, update_test_data, update_train_data):

    print  "\r",num_iter,
    ####### actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y})

    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        #print np.shape(batch_X)
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
        #   print np.shape(mnist.test.labels)
        test_a.append(a)
        test_c.append(c)


    return (train_a, train_c, test_a, test_c)


# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []
train_c = []
test_a = []
test_c = []

training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

# 7. Plot and visualise the accuracy and loss

# accuracy training vs testing dataset
plt.title("Accuracy: Training vs Test")
plt.plot(train_a,label="Training Accuracy")
plt.plot(test_a,label ="Test Accuracy")
plt.xlabel("epoch")
plt.legend(loc=4)
plt.grid(True)
plt.show()


# loss training vs testing dataset
plt.title("Loss: Training vs Test")
plt.plot(train_c,label="Training Loss")
plt.plot(test_c,label="Test Loss")
plt.xlabel("epoch")
plt.legend(loc=1)
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50
x_range = range(zoom_point,training_iter/epoch_size)
plt.title("Accuracy: Training vs Test")
plt.plot(x_range, train_a[zoom_point:],label="Training Accuracy")
plt.plot(x_range, test_a[zoom_point:],label="Test Accuracy")
plt.grid(True)
plt.xlabel("epoch")
plt.xticks(x_range)
plt.legend(loc=4)
plt.show()

plt.title("Loss: Training vs Test")
plt.plot(x_range,train_c[zoom_point:],label="Training Loss")
plt.plot(x_range,test_c[zoom_point:],label="Test Loss")
plt.grid(True)
plt.xlabel("epoch")
plt.xticks(x_range)
plt.legend(loc=1)
plt.show()

print("test accuracy from array  test_a: "+str( test_a[99]))
print("loss from array  test_c: "+str( test_c[99]))
