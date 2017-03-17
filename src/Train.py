""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import time                                 # to retreive current time

import BonaAge
import numpy as np
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run""")
tf.app.flags.DEFINE_integer('num_epochs', 10000, """How many epochs to run""")
tf.app.flags.DEFINE_integer('test_interval', 200, """How often to test the model during training""")
tf.app.flags.DEFINE_integer('print_interval', 200, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 1000, """How many steps to iterate before saving a checkpoint""")
tf.app.flags.DEFINE_integer('summary_steps', 1000, """How many steps to iterate before writing a summary""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Yes or no""")
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 0.01, """ The gamma value for regularization loss""")

# Define a custom training class
def train():
    """ Train our network for a number of steps
    The 'with' statement tells python to try and execute the following code, and utilize a custom defined __exit__
    function once it is done or it fails """

    tf.reset_default_graph()  # Makes this the default graph where all ops will be added

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get a dictionary of our images, id's, and labels here
    images = BonaAge.inputs()

    # Build a graph that computes the prediction from the inference model (Forward pass)
    logits, l2loss = BonaAge.forward_pass(images['image'], keep_prob=FLAGS.dropout_factor)

    # Make our final label the average of the two labels
    avg_label = tf.transpose(tf.divide(images['age'], 19))

    # Get some metrics
    predictions2 = tf.transpose(tf.multiply(logits, 19))
    labels2 = tf.transpose(tf.multiply(avg_label, 19))
    MAE_loss = tf.reduce_mean(tf.abs(tf.subtract(predictions2, labels2)))

    # Calculate the total loss, adding L2 regularization
    mse_loss = BonaAge.total_loss(logits, avg_label)

    loss = tf.add(mse_loss, l2loss, name='loss')

    # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
    train_op = BonaAge.backward_pass(loss, global_step)

    # Merge the summaries
    all_summaries = tf.summary.merge_all()

    # Initialize the handle to the summary writer in our training directory
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

    var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # SessionRunHook is called when you use a monitored session's run function after each run
    class _LoggerHook(tf.train.SessionRunHook):

        def begin(self):
            self._step = -1
            print('Starting Computations')

        def before_run(self, run_context):  # Called before each call to run()
            self._step += 1  # Increment step
            self._start_time = time.time()  # Start the timer for this iteration
            return tf.train.SessionRunArgs(loss)  # represents arguments to be added to the session.run call

        def after_run(self, run_context, run_values):  # Called after each call to run()
            duration = time.time() - self._start_time  # Calculate duration of each iteration
            loss_value = run_values.results * 100  # Get the current loss value To Do: Make average
            # Put if statements here for things you will do every x amount of steps
            if self._step % FLAGS.print_interval == 0:  # This statement will print loss, step and other stuff
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                format_str = ('Step %d, Loss: = %.4f (%.1f eg/s;)')
                print(format_str % (self._step, loss_value, examples_per_sec), end=" ")

                # Test the data
                predictions1, label1, loss1, mae = mon_sess.run([predictions2, labels2, mse_loss, MAE_loss])

                predictions = predictions1.astype(np.float)
                label = label1.astype(np.float)

                right = 0.0

                for i in range(0, predictions.size - 1):
                    if abs(predictions[i] - label[i]) < 0.833:  # If difference is less than 10 m
                        right += 1
                    elif abs(predictions[i] - label[i]) < 1.667:  # 2 standard deviations
                        right += 0.5

                acc = (right / (predictions.size - 1)) * 100

                np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
                print('Eg. Predictions: Network(Real): %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), '
                      'MSE: %.4f, MAE: %.4f, Accuracy: %.2f %%' % (predictions[0],
                                                                   label[0], predictions[1], label[1], predictions[2],
                                                                   label[2], predictions[3],
                                                                   label[3], loss1, mae, acc))

                # Run a session to retrieve our summaries
                summary = mon_sess.run(all_summaries)

                # Add the summaries to the protobuf for Tensorboard
                summary_writer.add_summary(summary, self._step)

                # if self._step % FLAGS.test_interval == 0:  # This statement will print loss, step and other stuff
                #     predictions, label = mon_sess.run([logits, avg_label])
                #     label *= 78
                #     predictions *= 78
                #     print ('Predictions: %s, Label: %s' %(predictions, label))

    # Creates a session initializer/restorer and hooks for checkpoint summary and saving
    # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                           hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                  tf.train.NanTensorHook(loss),
                                                  _LoggerHook()], save_checkpoint_secs=FLAGS.checkpoint_steps,
                                           save_summaries_steps=FLAGS.summary_steps,
                                           config=tf.ConfigProto(
                                               log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        # Initialize the variables
        mon_sess.run(var_init)

        # Initialize the enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

        try:
            while not mon_sess.should_stop():  # For the training coordinator. Only one thread here so we're good
                mon_sess.run(train_op)  # Runs the operations and evaluates tensors in train_op - One cycle/batch

        except tf.errors.OutOfRangeError:
            print('Done with Training - Epoch limit reached')
        finally:
            # Stop threads when done
            coord.request_stop()
            # Wait for threads to finish before closing session
            coord.join(threads)
            mon_sess.close()


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()