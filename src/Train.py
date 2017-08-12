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
tf.app.flags.DEFINE_integer('num_epochs', 900, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('model', 4, """1=YF, 2=OF, 3=YM, 4=OM""")

tf.app.flags.DEFINE_integer('cross_validations', 8, "X fold cross validation hyperparameter")
tf.app.flags.DEFINE_string('validation_file', 'test', "Which protocol buffer will be used fo validation")

# YG = 2309(152), OG: 4093(495), OM: 3259(500), YM: 1184(162), AM: 3636(697), AF: 5911(683)
tf.app.flags.DEFINE_integer('epoch_size', 5911, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 236, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 2365, """How many STEPS to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 25, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.3, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 2e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('learning_rate', 5e-5, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay', 0.98, """The base factor for exp learning rate decay""")
tf.app.flags.DEFINE_integer('lr_steps', 6000, """ The number of steps until we decay the learning rate""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
# All run settings: b1 = 0.9, b2 = 0.999, momentum = 0.9, Updater = Adam
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('momentum', 0.9, """ The momentum for the momentum optimizer""")
tf.app.flags.DEFINE_bool('use_nesterov', True, """ Whether to use nesterov""")


# Define a custom training class
def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Get a dictionary of our images, id's, and labels here
        images, validation = BonaAge.inputs(skip=True)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, l2loss = BonaAge.forward_pass(images['image'], phase_train1=True)

        # Make our ground truth the real age since the bone ages are normal
        avg_label = tf.transpose(tf.divide(images['reading'], 19))

        # Get some metrics
        predictions2 = tf.transpose(tf.multiply(logits, 19))
        labels2 = tf.transpose(tf.multiply(avg_label, 19))

        # Get MAE
        MAE = tf.metrics.mean_absolute_error(labels2, predictions2)

        # Make a summary of MAE
        tf.summary.scalar('MAE', MAE[1])

        # Calculate the objective function loss
        mse_loss = BonaAge.total_loss(logits, avg_label)

        # Add in L2 Regularization
        loss = tf.add(mse_loss, l2loss, name='loss')

        # Build the backprop graph to train the model with one batch and update the parameters (Backward pass)
        train_op = BonaAge.backward_pass(loss)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(0.999)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=5)

        # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
        with tf.Session() as mon_sess:

            # Retreive the checkpoint
            ckpt = tf.train.get_checkpoint_state('training/')

            # Initialize the variables
            mon_sess.run(var_init)

            if ckpt and ckpt.model_checkpoint_path:

                # Display message
                print ("Previous Checkpoint Found! Loading: %s" %ckpt.model_checkpoint_path)

                # Restore the learned variables
                restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                # Restore the graph
                restorer.restore(mon_sess, ckpt.model_checkpoint_path)

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, mon_sess.graph)

            # Initialize the thread coordinator
            coord = tf.train.Coordinator()

            # Start the queue runners
            threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

            # Initialize the step counter
            step = 0

            # Set the max step count
            max_steps = (FLAGS.epoch_size / FLAGS.batch_size) * FLAGS.num_epochs

            try:
                while step <= max_steps:

                    # Start the timer
                    start_time = time.time()

                    # Run an iteration
                    mon_sess.run(train_op)

                    # Calculate Duration
                    duration = time.time() - start_time

                    # Print interval code
                    if step % FLAGS.print_interval == 0:  # This statement will print loss, step and other stuff

                        # Get the MAE
                        predictions1, label1 = mon_sess.run([predictions2, labels2])

                        # Output the summary
                        predictions = predictions1.astype(np.float)
                        label = label1.astype(np.float)

                        # Calculate the accuracy
                        _, mae_batch = BonaAge.calculate_errors(predictions, label)

                        # Load some metrics for testing
                        predictions1, label1, loss1, loss2 = mon_sess.run([predictions2, labels2, mse_loss, l2loss])

                        # Output the summary
                        BonaAge.after_run(predictions1, label1, loss1, (loss2 * 100), step, duration, mae_batch)

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries)

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, step)

                    # Checkpoint saving step
                    if step % FLAGS.checkpoint_steps == 0:

                        # Save the checkpoint
                        print(" ---------------- SAVING CHECKPOINT MAE: %0.3f ------------------" % mae_batch)

                        # Define the filename
                        Epoch = int(step / (FLAGS.epoch_size / FLAGS.batch_size))
                        file = ('CkptMAE%0.3fEpoch%s' % (mae_batch, Epoch))

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join(FLAGS.train_dir, file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)


                    # Increment step
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

                # Save the final checkpoint
                print(" ---------------- SAVING FINAL CHECKPOINT ------------------ ")
                saver.save(mon_sess, 'training/CheckpointFinal')

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