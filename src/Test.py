""" Testing the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import BonaAge
import numpy as np
import tensorflow as tf
from Train import calculate_errors

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS


def evaluate():
    """Eval Accuracy for a number of steps."""
    with tf.Graph().as_default() as g:

        # Get images and labels for validation set
        images, validation, val_batches = BonaAge.inputs(skip=True)

        # Build a Graph that computes the logits predictions from the inference model.
        predictions1, _ = BonaAge.forward_pass(validation['image'], phase_train=False, bts=51)

        labels1 = validation['age']
        predictions2 = tf.transpose(tf.multiply(predictions1, 19))

        # Get the checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        # Variable initializer
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:

            sess.run(var_init)

            # Restore the learned variables for eval.
            saver = tf.train.import_meta_graph('training/Checkpoint.ckpt.meta')

            # Restore the saver
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                print('extending runners')
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                step = 0
                while step < 1 and not coord.should_stop():
                    predictions, label = sess.run([predictions2, labels1])

                    # Compute accuracy
                    predictions = predictions.astype(np.float)
                    label = label.astype(np.float)

                    # Calculate the accuracy
                    acc, mae = calculate_errors(predictions, label)

                    # Print the results
                    np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
                    print('Eg. Predictions: Network(Real): %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), '
                          'MAE: %.2f Yrs, Train Accuracy: %s %%'
                          % (predictions[0], label[0], predictions[1], label[1], predictions[2],
                             label[2], predictions[3], label[3], mae, acc))

                    step += 1

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            # while True:
            #   eval_once(saver, predictions, labels)
            #   if FLAGS.run_once:
            #     break
            #   time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
