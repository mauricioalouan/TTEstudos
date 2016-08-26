import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
	#This line uses TensorFlow’s transpose method, swapping the axes 0 and 1 around using the perm parameter (axis 2 stays where it is).
	x = tf.transpose(x, perm=[0, 1, 2])
	session.run(model)
	result = session.run(x)


plt.imshow(result)
plt.show()
