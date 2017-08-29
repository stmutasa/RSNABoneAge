""" Training the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import glob
import os
import time

import Competition
import numpy as np
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('dims', 256, "Size of the images")
tf.app.flags.DEFINE_integer('network_dims', 256, "Size of the images")
tf.app.flags.DEFINE_string('validation_file', '0', "Which protocol buffer will be used fo validation")
tf.app.flags.DEFINE_integer('cross_validations', 8, "X fold cross validation hyperparameter")

# Female = 852, Male = 990, YF: 434, OF: 494, YM 368
tf.app.flags.DEFINE_integer('epoch_size', 368, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Define a custom training class
def test():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Load the images and labels.
        _, validation = Competition.Inputs(skip=True)

        # Perform the forward pass:
        logits, _ = Competition.forward_pass_res(validation['image'], phase_train1=True)

        # Make our ground truth the real age since the bone ages are normal
        avg_label = tf.divide(validation['reading'], 19)
        # Get some metrics
        predictions2 = tf.multiply(logits, 19)
        labels2 = tf.multiply(avg_label, 19)

        # Output the summary of the MSE and MAE
        tf.summary.scalar('Absolute Error', tf.reduce_mean(tf.abs(tf.squeeze(avg_label) - tf.squeeze(logits))))

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize the handle to the summary writer in our training directory
        summary_writer = tf.summary.FileWriter('training/Test/')

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=5)

        # Track the top performers
        best_MAE = 4
        best_file = 'None'

        while True:

            # Check folder for changes
            filecheck = glob.glob('training/' + '*')

            # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

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
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                # Initialize the thread coordinator
                coord = tf.train.Coordinator()

                # Start the queue runners
                threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

                # Initialize the step counter
                step = 0

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Running values for accuracy calculation
                right, total, standard = 0, 0, 0

                try:
                    while step < max_steps:

                        # Retreive the predictions and labels
                        preds, labs = mon_sess.run([predictions2, labels2])

                        # Convert to numpy arrays
                        predictions = np.squeeze(preds.astype(np.float))
                        label = np.squeeze(labs.astype(np.float))

                        # Clip predictions
                        predictions[predictions < 0] = 0
                        predictions[predictions > 19] = 19

                        # Calculate MAE
                        MAE = np.mean(np.absolute(predictions - label))
                        STD = np.std(np.absolute(predictions - label))

                        # Print the summary
                        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
                        print('Pred: %s, STD: %.2f' % (predictions[:12], STD))
                        print('Real: %s, MAE: %.2f' % (label[:12], MAE))

                        # Append right
                        right += MAE
                        standard += STD
                        total += 1

                        # Garbage collection
                        del preds, labs, predictions, label, MAE, STD

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # Calculate final MAE and ACC
                    accuracy = right/total
                    deviation = standard/total

                    # Print the final accuracies and MAE
                    print('-' * 70)
                    print(
                        '--- EPOCH: %s MAE: %.4f, STD: %.3f (Old Best: %.2f - %s) ---'
                        % (Epoch, accuracy, deviation, best_MAE, best_file))

                    # Run a session to retrieve our summaries
                    summary = mon_sess.run(all_summaries)

                    # Add the summaries to the protobuf for Tensorboard
                    summary_writer.add_summary(summary, int(int(Epoch) * 47))

                    # Lets save runs below 0.8
                    if accuracy <= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filename
                        file = ('Epoch_%s_MAE_%0.3f' % (Epoch, accuracy))

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('testing/', file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Save a new best MAE
                        best_MAE = accuracy
                        best_file = Epoch

                    # Stop threads when done
                    coord.request_stop()

                    # Wait for threads to finish before closing session
                    coord.join(threads, stop_grace_period_secs=20)

                    # Shut down the session
                    mon_sess.close()

            # Break if this is the final checkpoint
            if 'Final' in Epoch: break

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            newfilec = glob.glob('training/' + '*')

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(int(FLAGS.epoch_size * 0.05))

                # Recheck the folder for changes
                newfilec = glob.glob('training/' + '*')


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists('testing/'):
        tf.gfile.DeleteRecursively('testing/')
    tf.gfile.MakeDirs('testing/')
    test()


if __name__ == '__main__':
    tf.app.run()