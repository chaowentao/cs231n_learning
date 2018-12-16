# Create TensorFlow object called tensor
import tensorflow as tf
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output.decode())# bytestring decode to string.