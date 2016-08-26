import tensorflow as tf

x = tf.constant(35, name='x')

y = tf.Variable(x + 5, name='y')

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    # Launch the graph in a session.
	# Create a summary writer, add the 'graph' to the event file.
    writer = tf.train.SummaryWriter("/tmp/basic", sess.graph)
    model = tf.initialize_all_variables()
    sess.run(model)
    print(sess.run(y))




#TensorBoard is a tool for visualizing the TensorFlow graph and analyzing recorded metrics during training and inference. The graph is created using the Python API, then written out using the tf.train.SummaryWriter.add_graph() method. When you load the file written by the SummaryWriter into TensorBoard, you can see the graph that was saved, and interactively explore it.

#However, TensorBoard is not a tool for building the graph itself. It does not have any support for adding nodes to the graph.