""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import time                                 # to retreive current time
from datetime import datetime  # Classes for manipulating the date and time displaying

import BonaAge
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('max_steps', 2000, """Number of batches to run""")
tf.app.flags.DEFINE_integer('num_epochs', 100, """How many epochs to run""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """How often to test the model during training""")
tf.app.flags.DEFINE_integer('print_interval', 10, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 100, """How many steps to iterate before saving a checkpoint""")
tf.app.flags.DEFINE_integer('summary_steps', 100, """How many steps to iterate before writing a summary""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Yes or no""")


# To do: Remove labels that are outside the normal range

# Define a custom training class
def train():
    """ Train our network for a number of steps
    The 'with' statement tells python to try and execute the following code, and utilize a custom defined __exit__
    function once it is done or it fails """
    with tf.Graph().as_default():        # Makes this the default graph where all ops will be added
        # Get the tensor that keeps track of step in this graph or create one if not there
        global_step = tf.contrib.framework.get_or_create_global_step()
        # Create a variable to count the number of train() calls equal to num of batches processed
        # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Get a dictionary of our images, id's, and labels here
        images = BonaAge.inputs(None)  # Set num epochs to none

        # Build a graph that computes the log odds unit prediction from the inference model (Forward pass)
        logits = BonaAge.forward_pass(images['image'])

        # Make our final label the average of the two labels
        avg_label = tf.divide(tf.add(images['label1'], images['label2']), 2)
        # avg_label = tf.cast(avg_label, tf.int32)  # For now define labels as integers

        # Calculate the total loss, adding L2 regularization and calculated cross entropy
        loss = BonaAge.total_loss(logits, avg_label)

        # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
        train_op = BonaAge.backward_pass(loss, global_step, True)

        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # SessionRunHook is called when you use a monitored session's run function after each run
        #To do: Define run_values and run_context
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1
                print('Starting Computations')

            def before_run(self, run_context):          # Called before each call to run()
                self._step +=1                          # Increment step
                self._start_time = time.time()          # Start the timer for this iteration
                return tf.train.SessionRunArgs(loss)    # represents arguments to be added to the session.run call

            def after_run(self, run_context, run_values):   #Called after each call to run()
                duration = time.time() - self._start_time   #Calculate duration of each iteration
                loss_value = run_values.results             #Get the current loss value To Do: Make average
                if self._step <= 1: print('Loss = %.3f' % loss_value)
                # Put if statements here for things you will do every x amount of steps
                if self._step % FLAGS.print_interval == 0:  # This statement will print loss, step and other stuff
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: Step %d, Loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

        # Creates a session initializer/restorer and hooks for checkpoint summary and saving
        # (master='', is_chief=True, checkpoint_dir, scaffold=None, hooks=None, chief_only_hooks=None,
        # save_checkpoint_secs=600, save_summaries_steps=100, config=None)
        # config Proto sets options for configuring the sessin like run on GPU, allocate GPU memory etc.
        # TO Do: Put back the stop on NAN once you fix the nana issue
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
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
                    mon_sess.run(
                        train_op)  # Runs the operations and evaluates tensors in train_op - One cycle for this batch

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')
            finally:
                # Stop threads when done
                coord.request_stop()
                # Wait for threads to finish before closing session
                coord.join(threads)
                #mon_sess.close()


def cst(coord=None, sess=None, threads=None):
    """ This function coordinates the execution of the graph"""
    if coord is None:
        coord = tf.train.Coordinator()  # Coordinates the timing and termination of threads
        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])  # Runs one "step"
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # Starts the queue runners in the graph
        return coord, sess, threads
    else:
        coord.request_stop()  # Request a stop of threads. should_stop() will return True
        coord.join(threads)  # Waits for threads to terminate
        sess.close()
        return


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()