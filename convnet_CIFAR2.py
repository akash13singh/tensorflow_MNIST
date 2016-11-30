#source for data reading: https://github.com/kgeorge/kgeorge_dpl/blob/master/notebooks/tf_cifar.ipynb
from __future__ import division
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
import os
import numpy as np
import time


tf.set_random_seed(0)


def load_cifar(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as inf:
        cifar = pickle.load(inf)
    data = cifar['data'].reshape((10000, 3, 32, 32))
    data = np.rollaxis(data, 3, 1)
    data = np.rollaxis(data, 3, 1)
    y = np.array(cifar['labels'])
    # print data.shape
    # print y.shape
    return data, y

# read training/etst data

train_X = np.empty(shape=(50000,32,32,3))
train_Y = np.empty(shape=(50000))
test_X = np.empty(shape=(10000,32,32,3))
test_Y = np.zeros(shape=(10000))
y_tmp = np.empty(shape=(10000))
path ="cifar-10-batches-py"

i=0
for (dirpath, dirnames, filenames)  in os.walk(path):
    for file in filenames:
        if file.startswith("data"):
            # print file
            (train_X[i:i+10000,],train_Y[i:i+10000]) =load_cifar(path+"/"+file)
            # for idx,value in enumerate(y_tmp):
            #     train_Y[i+idx,value] =1
            # print train_Y[i:i+2]
            # print y_tmp[:2]
            i=i+10000
        if file.startswith("test"):
            # print file
            (test_X[0:10000,],test_Y[0:10000]) =load_cifar(path+"/"+file)
            # for idx, value in enumerate(y_tmp):
            #     test_Y[idx, value] = 1
            # print test_Y[0: 2]
            # print y_tmp[:2]

# train_X = train_X[0:1000,]
# train_Y = train_Y[0:1000,]
# test_X = test_X[0:100,]
# test_Y = test_Y[0:100,]




# 1. Define Variables and Placeholders
# Model is Y = softmax(X*W + b)

# input f 32 X 32 images with 3 channel (color images) . None is left for minibatch size
X = tf.placeholder(tf.float32,[None,32,32,3])


# Y_ true labels. One-Hot Encoding
Y_= tf.placeholder(tf.float32,[None,10])

def create_fully_connected_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def create_conv_weight(patch_height,patch_width,input_channel,output_channel):
  initial = tf.truncated_normal(shape=[patch_height,patch_width,input_channel,output_channel], stddev=0.1)
  return tf.Variable(initial)

def create_bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def create_strides(batch_step, height_step, width_step, channel_step):
    return [batch_step, height_step, width_step, channel_step]

def create_conv_layer(input, W,strides,padding='SAME'):
  return tf.nn.conv2d(input, W, strides, padding)

def apply_max_pool(x,ksize,strides, padding='SAME'):
  return tf.nn.max_pool(x, ksize,strides, padding)


# #Parameters
#create first conv layer, with 3 input channel of orig. image, 4 output channels, stride of 1*1 and padding =SAME
W1 = create_conv_weight(5,5,3,4)
print W1.get_shape()
B1 = create_bias([4])
strides1 = create_strides(1,1,1,1)
Y1 = tf.nn.relu(create_conv_layer(X,W1,strides1,padding="SAME")+B1)
print (Y1.get_shape())

W2 = create_conv_weight(5,5,4,8)
B2 = create_bias([8])
strides2 = create_strides(1,2,2,1)
Y2 = tf.nn.relu(create_conv_layer(Y1,W2,strides2,padding="SAME")+B2)


W3 = create_conv_weight(4,4,8,12)
B3 = create_bias([12])
strides3 = create_strides(1,2,2,1)
Y3 = tf.nn.relu(create_conv_layer(Y2,W3,strides3,padding="SAME")+B3)

# print (type(Y3))
# print (np.shape(Y3))
keep_prob = tf.placeholder(tf.float32)
# Y3_drop = tf.nn.dropout(Y3,keep_prob)

Y3_reshaped = tf.reshape(Y3,[-1,8*8*12])
W4 = create_fully_connected_weight([8*8*12,200])
B4 = create_bias([200])
Y4 = tf.nn.relu(tf.matmul(Y3_reshaped,W4)+B4)
Y4_drop = tf.nn.dropout(Y4,keep_prob=keep_prob)

W5 = create_fully_connected_weight([200,10])
B5 = create_bias([10])
Ylogits = tf.matmul(Y4_drop,W5)+B5


# 3. Define the loss function
cross_entropy =  tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels= Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

#prediction
correct_prediction = tf.equal(tf.argmax(Ylogits,1),tf.argmax(Y_,1))

# 4. Define the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(.005).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(.01,0.9).minimize(cross_entropy)
#step = tf.Variable(0,trainable=False)
#train_step = tf.train.MomentumOptimizer(.01,.9).minimize(cross_entropy,global_step=step)


# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


n_classes=10
batch_size=250
n_epochs=40
n_batches_train = int(train_Y.shape[0]//batch_size)
print "number of batches: %d"%(n_batches_train)

def all_batches_run_train(n_batches, data=None, labels=None):
    sum_all_batches_loss = 0
    sum_all_batches_acc = 0
    sum_n_samples = 0
    for b in xrange(n_batches):
        offset = b * batch_size
        batch_data = data[offset: offset + batch_size, :, :, :]
        n_samples = batch_data.shape[0]
        #print('hello here n_samples =%d' % n_samples)
        batch_labels = labels[offset: offset + batch_size]
        batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
        # print np.shape(batch_data)
        # print np.shape(batch_labels)
        feed_dict = {X: batch_data,Y_: batch_labels,keep_prob:0.5}
        _, loss_value, a = sess.run([train_step, cross_entropy, accuracy], feed_dict=feed_dict)
        sum_all_batches_loss += loss_value * n_samples
        sum_all_batches_acc += a * n_samples
        sum_n_samples += n_samples
        if (n_samples != batch_size):
            print('n_samples =%d' % n_samples)
    print "sum of samples trained %d" %(sum_n_samples)
    return (sum_all_batches_loss / sum_n_samples, sum_all_batches_acc / sum_n_samples)


def run_test(data=None, labels=None):
    assert (data.shape[0] == labels.shape[0])
    batch_size_test = 10000
    labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    feed_dict = {X: data, Y_: labels,keep_prob:1}
    test_a = sess.run([accuracy], feed_dict=feed_dict)
    return test_a

i=1

train_ac = []
train_loss = []
test_ac = []
for e in xrange(n_epochs):
    start_time = time.time()
    n_data = train_X.shape[0]
    #print n_data
    perm = np.random.permutation(n_data)
    train_X = train_X[perm, :, :, :]
    train_Y = train_Y[perm]
    mean_loss_per_sample_train, accuracy_per_sample_train = all_batches_run_train(n_batches_train, data=train_X,labels=train_Y)
    test_a = run_test(data=test_X, labels=test_Y)
    print "loss after epoch %d = %f: "%(i,mean_loss_per_sample_train)
    print "train accuracy after epoch %d = %f: " % (i, accuracy_per_sample_train)
    print "test accuracy after epoch %d = %f: " % (i, test_a[0])

    i=i+1
    train_ac.append(accuracy_per_sample_train)
    train_loss.append(mean_loss_per_sample_train)
    test_ac.append(test_a[0])


print('done training')

plt.title("Training Accuracy over epochs")
plt.plot(train_ac,label="Training Accuracy")
plt.plot(test_ac,label="Test Accuracy")
plt.xlabel("epoch")
plt.legend(loc=4)
plt.grid(True)
plt.show()

plt.title("Training loss over epochs")
plt.plot(train_loss,label="Training Loss")
plt.xlabel("epoch")
plt.grid(True)
plt.show()


test_a = run_test(data=test_X, labels=test_Y)
print('done testing')
print("Testing Accuracy "+str(test_a[0]))


