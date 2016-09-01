import tensorflow as tf
#download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation. 
#We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. 
x = tf.placeholder(tf.float32, [None, 784]) #symbolic variables
#tensors full of zeros
#we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes.
W = tf.Variable(tf.zeros([784, 10]))#weights
#b has a shape of [10] so we can add it to the output.
b = tf.Variable(tf.zeros([10]))#biases
#First, we multiply x by W with the expression tf.matmul(x, W), we then add b, and finally apply tf.nn.softmax.
y = tf.nn.softmax(tf.matmul(x, W) + b)
#To implement cross-entropy we need to first add a new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#the cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#trainning steps(In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5.)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#create an operation to initialize the variables we created.
init = tf.initialize_all_variables()
#We can now launch the model in a Session, and now we run the operation that initializes the variables
sess = tf.Session()
sess.run(init)
#Let's train -- we'll run the training step 1000 times
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 #Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.
 #-------------------EVALUATING THE MODEL---------------------------------- 
#while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Finally, we ask for our accuracy on our test data.
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#print(sess.run(correct_prediction, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))