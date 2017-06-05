""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time

import BonaAge
import numpy as np
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
# Train and validation set sizes: YG: 206/51, OG: 340/85, OM: 346/86
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('model', 1, """1 Y=F, 2=OF, 3=YM, 4=OM""")
tf.app.flags.DEFINE_integer('num_epochs', 1, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('epoch_size', 32, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 1, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('validation_file', 3, "Which protocol buffer will be used fo validation")
tf.app.flags.DEFINE_integer('test_file', 4, "Which protocol buffer will be used for testing")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")

# Define a custom training class
def test():
    """ Train our network for a number of steps
    The 'with' statement tells python to try and execute the following code, and utilize a custom defined __exit__
    function once it is done or it fails """

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Get a dictionary of our images, id's, and labels here
        images, validation, val_batch = BonaAge.inputs(skip=True)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BonaAge.forward_pass(validation['image'], phase_train1=False)

        # Make our ground truth the real age since the bone ages are normal
        avg_label = tf.divide(tf.add(validation['label1'], validation['label2']), 2)

        # Get some metrics
        predictions2 = tf.transpose(tf.multiply(logits, 19))
        labels2 = avg_label

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Initialize the saver
        saver = tf.train.Saver()

        while True:

            with tf.Session() as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state('training/')

                # Initialize the variables
                mon_sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(mon_sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1]

                # Initialize the thread coordinator
                coord = tf.train.Coordinator()

                # Start the queue runners
                threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = (FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs

                # Running values for MAE and Acc
                total_MAE = []
                total_ACC = []

                try:
                    while step <= max_steps:

                        # Load some metrics for testing
                        predictions1, label1 = mon_sess.run([predictions2, labels2])

                        # Output the summary
                        predictions = predictions1.astype(np.float)
                        label = label1.astype(np.float)

                        # Calculate the accuracy
                        acc, mae = BonaAge.calculate_errors(predictions, label)

                        # Append the values
                        total_ACC.append(acc)
                        total_MAE.append(mae)

                        # Print the summary
                        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
                        # print('Eg. Predictions: Network(Real): %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), '
                        #       'MAE: %.2f Yrs, Train Accuracy: %s %%'
                        #       % (predictions[0], label[0], predictions[1], label[1], predictions[2],
                        #          label[2], predictions[3], label[3], mae, acc))

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    accuracy = sum(total_ACC) / len(total_ACC)
                    Mean_AE = float(sum(total_MAE) / len(total_MAE))

                    # Print the final accuracies and MAE
                    print('-' * 70)
                    print(
                        '--------- EPOCH: %s TOTAL MAE: %.3f, TOTAL ACCURACY: %.2f %% -------' % (Epoch, Mean_AE, accuracy))

                    # Stop threads when done
                    coord.request_stop()

                    # Wait for threads to finish before closing session
                    coord.join(threads, stop_grace_period_secs=20)

                    # Shut down the session
                    mon_sess.close()

            print ('Sleeping')
            time.sleep(60)


def main(argv=None):  # pylint: disable=unused-argument
    test()


if __name__ == '__main__':
    tf.app.run()
