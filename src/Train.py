""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time

import BonaAge
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
                        BonaAge.after_run(predictions1, label1, loss1, loss2, step, duration)

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


def main(argv=None):  # pylint: disable=unused-argument
    train()

if __name__ == '__main__':
    tf.app.run()