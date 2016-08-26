#Using the code from (2) and (3) above, 
#create a program that computers the “rolling” average of the 
#following line of code: np.random.randint(1000). In other words, 
#keep looping, and in each loop, call np.random.randint(1000) once 
#in that loop, and store the current average in a Variable that keeps 
#updating each loop.
import tensorflow as tf
import numpy as np
i=[]	
x = np.random.randint(10, size=100)


model = tf.initialize_all_variables()


with tf.Session() as session:
    for i in x:
        session.run(model)
        i = i + 1
        print(i)
