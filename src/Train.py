""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time

import BonaAge
import numpy as np
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('num_epochs', 500, """Number of epochs to run""")
# Young girls = 206 (51),
tf.app.flags.DEFINE_integer('epoch_size', 206, """How many images were loaded""")
tf.app.flags.DEFINE_integer('test_interval', 650, """How often to test the model during training""")
tf.app.flags.DEFINE_integer('print_interval', 130, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 4000, """How many STEPS to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")

# Hyperparameters:
# For the old girls run: lr = .001, dropout = 0.5, gamma = 0.001, moving decay = 0.999, lr decay: 0.95, steps = 130
# Young girls run:l2 = 0.001, lr = 0.001, moving decay = 0.999, dropout = 1. beta: 0.9 and 0.999: 100% % 90k
# Old male run: l2 = 0.001, lr = 0.001, moving decay = 0.999, dropout = 0.5. lr decay 0.99, lr steps 200
# young male run: l2 = 0.001, lr = 0.001, moving decay = 0.999, dropout = 0.5. lr decay 0.99, lr steps 200
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 0.001, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay', 0.99, """The base factor for exp learning rate decay""")
tf.app.flags.DEFINE_integer('lr_steps', 200, """ The number of steps until we decay the learning rate""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
# All run settings: b1 = 0.9, b2 = 0.999, momentum = 0.9, Updater = Adam
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('momentum', 0.9, """ The momentum for the momentum optimizer""")
tf.app.flags.DEFINE_bool('use_nesterov', True, """ Whether to use nesterov""")


# Define a custom training class
def train():
    """ Train our network for a number of steps
    The 'with' statement tells python to try and execute the following code, and utilize a custom defined __exit__
    function once it is done or it fails """

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Get a dictionary of our images, id's, and labels here
        images, validation, val_batch = BonaAge.inputs(skip=False)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BonaAge.forward_pass(images['image'], phase_train=True)

        # Make our ground truth the real age since the bone ages are normal
        avg_label = tf.transpose(tf.divide(images['age'], 19))

        # Get some metrics
        predictions2 = tf.transpose(tf.multiply(logits, 19))
        labels2 = tf.transpose(tf.multiply(avg_label, 19))

        # Calculate the objective function loss
        mse_loss = BonaAge.total_loss(logits, avg_label)

        # Add in L2 Regularization
        loss = tf.add(mse_loss, l2loss, name='loss')

        # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
        train_op = BonaAge.backward_pass(loss)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize the handle to the summary writer in our training directory
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Get the checkpoint
        # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        # Initialize the saver
        saver = tf.train.Saver()

        # Initialize the restorer
        # restorer = tf.train.import_meta_graph('training/Checkpoint.ckpt.meta')

        # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
        with tf.Session() as mon_sess:

            # Restore the saver
            # if ckpt: restorer.restore(mon_sess, ckpt.model_checkpoint_path)

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize the thread coordinator
            coord = tf.train.Coordinator()

            # Start the queue runners
            threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

            # Initialize the step counter
            step = 0

            # Set the max step count
            max_steps = (FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs

            # Define the checkpoint file:
            checkpoint_file = os.path.join(FLAGS.train_dir, 'Checkpoint.ckpt')

            try:
                while step <= max_steps:
                    start_time = time.time()  # Start the timer for this iteration
                    mon_sess.run(train_op)  # One iteration
                    duration = time.time() - start_time  # Calculate duration of each iteration
                    step += 1

                    # Put if statements here for things you will do every x amount of steps
                    if step % FLAGS.checkpoint_steps == 0:
                        # Save the checkpoint
                        print(" ---------------- SAVING CHECKPOINT ------------------")
                        saver.save(mon_sess, checkpoint_file)

                    if step % FLAGS.print_interval == 0:  # This statement will print loss, step and other stuff

                        # Load some metrics for testing
                        predictions1, label1, loss1, loss2 = mon_sess.run([predictions2, labels2, mse_loss, loss])

                        # Output the summary
                        after_run(predictions1, label1, loss1, loss2, step, duration)

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries)

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, step)

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

                # Save the final checkpoint
                print(" ---------------- SAVING FINAL CHECKPOINT ------------------ ")
                saver.save(mon_sess, checkpoint_file)

                # Stop threads when done
                coord.request_stop()

                # Wait for threads to finish before closing session
                coord.join(threads, stop_grace_period_secs=60)

                # Shut down the session
                mon_sess.close()


def RunningMean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def calculate_errors(predictions, label, Girls=True):
    """
    This function retreives the labels and predictions and then outputs the accuracy based on the actual
    standard deviations from the atlas of bone ages. The prediction is considered "right" if it's within
    two standard deviations
    :param predictions:
    :param labels:
    :param girls: Whether we're using the female or male standard deviations
    :return: Accurace : calculated as % of right/total
    """

    # First define our variables:
    right = 0.0  # Number of correct predictions
    total = predictions.size  # Number of total predictions
    std_dev = np.zeros_like(predictions, dtype='float32')  # The array that will hold our STD Deviations
    tot_err = 0.0

    # No apply the standard deviations TODO: Boys have different ranges lol
    for i in range(0, total):

        # Bunch of if statements assigning the STD for the patient's true age
        if label[i] <= (3 / 12):
            std_dev[i] = 0.72 / 12
        elif label[i] <= (6 / 12):
            std_dev[i] = 1.16 / 12
        elif label[i] <= (9 / 12):
            std_dev[i] = 1.36 / 12
        elif label[i] <= (12 / 12):
            std_dev[i] = 1.77 / 12
        elif label[i] <= (18 / 12):
            std_dev[i] = 3.49 / 12
        elif label[i] <= (24 / 12):
            std_dev[i] = 4.64 / 12
        elif label[i] <= (30 / 12):
            std_dev[i] = 5.37 / 12
        elif label[i] <= 3:
            std_dev[i] = 5.97 / 12
        elif label[i] <= 3.5:
            std_dev[i] = 7.48 / 12
        elif label[i] <= 4:
            std_dev[i] = 8.98 / 12
        elif label[i] <= 4.5:
            std_dev[i] = 10.73 / 12
        elif label[i] <= 5:
            std_dev[i] = 11.65 / 12
        elif label[i] <= 6:
            std_dev[i] = 10.23 / 12
        elif label[i] <= 7:
            std_dev[i] = 9.64 / 12
        elif label[i] <= 8:
            std_dev[i] = 10.23 / 12
        elif label[i] <= 9:
            std_dev[i] = 10.74 / 12
        elif label[i] <= 10:
            std_dev[i] = 11.73 / 12
        elif label[i] <= 11:
            std_dev[i] = 11.94 / 12
        elif label[i] <= 12:
            std_dev[i] = 10.24 / 12
        elif label[i] <= 13:
            std_dev[i] = 10.67 / 12
        elif label[i] <= 14:
            std_dev[i] = 11.3 / 12
        elif label[i] <= 15:
            std_dev[i] = 9.23 / 12
        else:
            std_dev[i] = 7.31 / 12

        # Calculate the MAE
        abs_err = abs(predictions[i] - label[i])
        tot_err += abs_err

        # Mark it right if we are within 2 std_devs
        if abs_err <= (std_dev[i] * 2):  # If difference is less than 2 stddev
            right += 1

    accuracy = (right / total) * 100  # Calculate the percent correct
    mae = (tot_err / total)

    return accuracy, mae


def after_run(predictions1, label1, loss1, loss_value, step, duration):
    # First print the number of examples per step
    eg_s = FLAGS.batch_size / duration
    print('Step %d, Loss: = %.2f (%.1f eg/s;)' % (step, loss_value, eg_s), end=" ")

    predictions = predictions1.astype(np.float)
    label = label1.astype(np.float)

    # Calculate the accuracy
    acc, mae = calculate_errors(predictions, label)

    # Print the summary
    np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
    print('Eg. Predictions: Network(Real): %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), '
          'MSE: %.4f, MAE: %.2f Yrs, Train Accuracy: %s %%'
          % (predictions[0], label[0], predictions[1], label[1], predictions[2],
             label[2], predictions[3], label[3], loss1, mae, acc))

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()