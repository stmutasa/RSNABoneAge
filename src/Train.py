""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

from datetime import datetime               # Classes for manipulating the date and time displaying
import time                                 # to retreive current time
import tensorflow as tf
import BonaAge

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'Training', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """How often to test the model during training""")
tf.app.flags.DEFINE_integer('print_interval', 10, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 100, """How many steps to iterate before saving a checkpoint""")
tf.app.flags.DEFINE_integer('summary_steps', 100, """How many steps to iterate before writing a summary""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Yes or no""")

# Define a custom training class
def train():
    """ Train our network for a number of steps
    The 'with' statement tells python to try and execute the following code, and utilize a custom defined __exit__
    function once it is done or it fails """
    with tf.Graph().as_default():        # Makes this the default graph where all ops will be added
        # Get the tensor that keeps track of step in this graph or create one if not there
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get the images and labels for our data set here
        # To do images, labels = BonaAge.processed_inputs():

        # Build a graph that computes the log odds unit prediction from the inference model (Forward pass)
        logits = BonaAge.forward_pass (images)

        # To do : Calculate the total loss, adding L2 regularization and calculated cross entropy
        loss = BonaAge.total_loss(logits, labels)

        # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
        train_op = BonaAge.backward_pass(loss, global_step)

        # SessionRunHook is called when you use a monitored session's run function after each run
        #To do: Define run_values and run_context
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1

            def before_run(self, run_context):          # Called before each call to run()
                self._step +=1                          # Increment step
                self._start_time = time.time()          # Start the timer for this iteration
                return tf.train.SessionRunArgs(loss)    # represents arguments to be added to the session.run call

            def after_run(self, run_context, run_values):   #Called after each call to run()
                duration = time.time() - self._start_time   #Calculate duration of each iteration
                loss_value = run_values.results             #Get the current loss value To Do: Make average
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
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                           hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                  tf.train.NanTensorHook(loss),
                                                  _LoggerHook()],save_checkpoint_secs=FLAGS.checkpoint_steps,
                                           save_summaries_steps=FLAGS.summary_steps,
                                           config=tf.configProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():   # For the training coordinator. Only one thread here so we're good
            mon_sess.run(train_op)          # Runs the operations and evaluates tensors in train_op - One cycle for this batch

# What does this shit do? Who knows, but make sure it's in there or the code won't work

def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()