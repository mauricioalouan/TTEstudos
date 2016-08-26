#Generate a NumPy array of 10,000 random numbers (called x) 
#and create a Variable storing the equation y=5(x^2)−3x+15
import tensorflow as tf
import numpy as np
x = np.random.randint(1000, size=10000)
y = tf.Variable(5*(x^2)-3*x+15, name='y')


model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))
