from . import hide_warnings
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_SIZE = 784
CODE_SIZE = 20
BATCH_SIZE = 100

# Statistics the normal distribution should have
MEAN = 0.0
STDEV = 1.0


def show_image(x, name="image"):
    # This assumes the input is a square image...
    width_height = int(INPUT_SIZE**0.5)
    img = tf.reshape(x, [-1, width_height, width_height, 1])
    tf.summary.image(name, img, max_outputs=1)

def mask(inputs):
    mask_ = np.full_like(inputs, 0.5)
    for i, x in enumerate(mask_):
        curr = 0.0
        for j, _ in enumerate(x):
            mask_[i][j] = curr
            curr += 1.0 / INPUT_SIZE
    return inputs * mask_

if __name__ == "__main__":
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)

    inputs = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    show_image(inputs, "inputs")
    with tf.name_scope("encoding"):
        code = fully_connected(inputs, CODE_SIZE)
    with tf.name_scope("decoding"):
        outputs = fully_connected(code, INPUT_SIZE)
    show_image(outputs, "outputs")
    with tf.name_scope("optimizer"):
        mse = tf.losses.mean_squared_error(inputs, outputs)
        tf.summary.scalar("mse", mse)
        # mean, var = tf.nn.moments(code, axis=[0])
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log", sess.graph)
        tf.global_variables_initializer().run()

        for i in range(10000):
            batch_inputs, _ = mnist.train.next_batch(BATCH_SIZE)
            if i % 2 == 0:
                batch_inputs = mask(batch_inputs)
            summary, _ = sess.run([merged, optimizer], feed_dict={
                inputs: batch_inputs
                })
            writer.add_summary(summary, i)