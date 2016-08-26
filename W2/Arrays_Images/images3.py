#ThThis is a flip (left-right), swapping the pixels from one side to another. 
#TensorFlow has a method for this called reverse_sequence, but the signature is a bit odd. 


import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "Teste.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
#It iterates over the image top to bottom (along its height), and slices left to right (along its width). From here, it then takes a slice of size width, where width is the width of the image.
#The code np.ones((height,)) * width creates a NumPy array filled with the value width. This is not very efficient! Unfortunately, at time of writing, it doesn’t appear that this function allows you to specify just a single value.
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()

#Iterate through the array according to batch_dim. Setting batch_dim=0 means we go through the rows (top to bottom).
#For each item in the iteration
#Slice a second dimension, denoted by seq_dim. Setting seq_dim=1 means we go through the columns (left to right).
#The slice for the nth item in the iteration is denoted by the nth item in seq_lengths