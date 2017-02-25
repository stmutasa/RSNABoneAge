""" Evaluation program for the network. Tests our predictions """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math  # To handle division
import time
from datetime import datetime  # To handle timing

import BonaAge
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'Test_Logs', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'training', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10, """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 6, """Number of examples to test.""")
tf.app.flags.DEFINE_boolean('run_once', True, """Whether to run eval only once.""")


def test_once(saver, MSE1, MSE2, summary_op):
    """ This function runs one iteration of testing data and evaluates the accuracy
    Args:
        saver: The saver used to restore the training dat
        summary_writer: The saver used to save a summary of the testing accuracy
        top_k_op: top_k_op
        summary_op: The operation to pass to sess.run()"""

    # Initialize the session
    with tf.Session() as sess:

        # Load the checkpoint state protobuff from the file. Returns none if there is nothign
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        # Restore if the checkpoint exists
        if checkpoint and checkpoint.model_checkpoint_path:
            print('Checkpoint Exists!')
            # Restore the previoiusly saved variables. Initializes them automatically
            saver.restore(sess, checkpoint.model_checkpoint_path)

            # Extract the global step from the model. Assuming the path is src/training/checkpoint
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]

        else:
            print('No checkpoint file found')  # Self explanatory
            return

        # Start the queue runners and enqueue threads
        coord = tf.train.Coordinator()  # Contains multiple threads
        print('Started Queue Runners')

        # Do the whole shebang
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):  # retrieve all of the queue runners

                # Create threads to run enqueue ops and start them. Daemon threads allow exiting without closing
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))  # Rounds up to nearest integer
                total_MSE = 0  # Counts the number of correct predictions
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                # Calculate the predictions TO Do change for MSE logistic regression
                while step < num_iter and not coord.should_stop():
                    print('Retrieving MSE Tensors...')
                    predictions = sess.run([MSE1, MSE2])
                    total_MSE += np.sum(predictions)
                    print('Total MSE @ step %s: %f' % (step, total_MSE))
                    step += 1
                    if step % 3 == 0: print('Another 3 steps')

                # Compute the Mean Squared Error
                MSE_avg = total_MSE / total_sample_count
                print('%s: Mean Squared Error @ %s: = %0.4f' % (datetime.now(), step, MSE_avg))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=60)


def test():
    """ Test our network"""

    with tf.Graph().as_default() as graph:

        # First load the images and labels
        test_data = FLAGS.eval_data == 'test'

        # Load our dictionary of images and labels
        image_dict = BonaAge.inputs(None)

        # Perform a forward pass with the data
        predicted = BonaAge.forward_pass(image_dict['image'])

        # Restore the predicted value to the units we used for the labels initially
        predicted = predicted * 38 / 2

        # We now have predicted (calculated values) and know the labels under image_dict['labelx']
        MSE1 = tf.reduce_mean(tf.square(image_dict['label1'] - predicted))
        MSE2 = tf.reduce_mean(tf.square(image_dict['label2'] - predicted))

        # Restore the moving average version of the learned variables for eval
        # variable_average = tf.train.ExponentialMovingAverage(BonaAge.MOVING_AVERAGE_DECAY)
        # restore_variables = variable_average.variables_to_restore()
        saver = tf.train.Saver()

        # Buid the summary operation based on the TF collection of summaries
        summary_op = tf.summary.merge_all()

        while True:
            test_once(saver, MSE1, MSE2, summary_op)
            if FLAGS.run_once: break
            time.sleep(FLAGS.eval_interval_secs)



def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
