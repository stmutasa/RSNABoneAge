""" Evaluation program for the network. Tests our predictions """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math  # To handle division
from datetime import datetime  # To handle timing

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'Test_Logs', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'training', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10, """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 100, """Number of examples to test.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only once.""")


def test_once(saver, summary_writer, top_k_op, summary_op):
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
            # Restore the previoiusly saved variables. Initializes them automatically
            saver.restore(sess, checkpoint.model_checkpoint_path)

            # Extract the global step from the model. Assuming the path is src/training/checkpoint
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]

        else:
            print('No checkpoint file found')  # Self explanatory
            return

        # Start the queue runners and enqueue threads
        coord = tf.train.Coordinator()  # Contains multiple threads

        # Do the whole shebang
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):  # retrieve all of the queue runners

                # Create threads to run enqueue ops and start them. Daemon threads allow exiting without closing
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))  # Rounds up to nearest integer
                true_count = 0  # Counts the number of correct predictions
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0

                # Calculate the predictions TO Do change for MSE logistic regression
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                # Compute the Mean Squared Error
                MSE = true_count / total_sample_count
                print('%s: Mean Squared Error @ %s: = %0.4f' % (datetime.now(), step, MSE))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=60)


def test():
    return


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
