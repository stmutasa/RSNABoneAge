"""
This folder is for the RSNA bone age competition
To keep it separate.

Train
"""

import os
import time

import Competition
import numpy as np
import tensorflow as tf

# Define flags
FLAGS = tf.app.flags.FLAGS

# Global variables
tf.app.flags.DEFINE_integer('dims', 256, "Size of the images")
tf.app.flags.DEFINE_integer('network_dims', 256, "Size of the images")
tf.app.flags.DEFINE_string('validation_file', '0', "Which protocol buffer will be used fo validation")
tf.app.flags.DEFINE_integer('cross_validations', 8, "X fold cross validation hyperparameter")

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('num_epochs', 1200, """Number of epochs to run""")
tf.app.flags.DEFINE_string('gender', 'M', """Which version to run""")

# Female = 5958, 950 @ 64, Male = 6934, 108 @ 64, YF: 3036, 47,
tf.app.flags.DEFINE_integer('epoch_size', 3036, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 94, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 945, """How many STEPS to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.7, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate',1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay', 0.98, """The base factor for exp learning rate decay""")
tf.app.flags.DEFINE_integer('lr_steps', 3000, """ The number of steps until we decay the learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('loss_factor', 0.0, """Addnl. fac. for the cost sensitive loss (2 makes 0 == 3x more)""")


def train():

    # First, make this the default graph where all ops will be added
    with tf.Graph().as_default():

        # Load the images and labels.
        data, _ = Competition.Inputs(skip=True)

        # Perform the forward pass:
        logits, l2loss = Competition.forward_pass_res(data['image'], phase_train1=True)

        # Make our ground truth the real age since the bone ages are normal
        avg_label = tf.divide(data['reading'], 19)
        #avg_label = data['reading']

        # Get some metrics
        predictions2 = tf.multiply(logits, 19)
        labels2 = tf.multiply(avg_label, 19)
        # predictions2 = logits
        # labels2 = avg_label

        # Get MAE
        MAE = tf.metrics.mean_absolute_error(labels2, predictions2)

        # Make a summary of MAE
        tf.summary.scalar('MAE', MAE[1])

        # Calculate the SCE loss. (softmax cross entropy with logits)
        #MSE_loss = Competition.total_loss(logits, data['reading'])
        MSE_loss = Competition.total_loss(logits, avg_label)

        # Add the L2 regularization loss
        loss = tf.add(MSE_loss, l2loss, name='TotalLoss')

        # Retreive the training operation with the applied gradients
        train_op = Competition.backward_pass(loss)

        # -------------------  Housekeeping functions  ----------------------

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # -------------------  Session Initializer  ----------------------

        # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
        with tf.Session() as mon_sess:

            # Initialize the variables
            mon_sess.run(var_init)

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

                    # Console and Tensorboard print interval
                    if step % FLAGS.print_interval == 0:

                        # How long did this take?
                        duration = time.time() - start_time

                        # First retreive the loss values
                        l2, sce, tot = mon_sess.run([l2loss, MSE_loss, loss])

                        # Also retreive the predictions and labels
                        preds, labs = mon_sess.run([predictions2, labels2])

                        # Convert to numpy arrays
                        predictions = np.squeeze(preds.astype(np.float))
                        label = np.squeeze(labs.astype(np.float))

                        # Clip predictions
                        predictions[predictions<0] = 0
                        predictions[predictions>19] = 19

                        # Calculate MAE
                        MAE = np.mean(np.absolute((predictions - label)))

                        # Now print the loss values
                        print ('-'*70)
                        print('Step: %s, Time Elapsed: %.1f sec, L2 Loss: %.4f, MSE: %.4f, Total Loss: %.4f'
                              % (step, duration, l2, sce, tot))

                        # Print the summary
                        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
                        print('Pred: %s' % predictions[:12])
                        print('Real: %s, MAE: %s' % (label[:12], MAE))

                        # Run a session to retrieve our summaries
                        summary = mon_sess.run(all_summaries)

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, step)

                    if step % FLAGS.checkpoint_steps == 0:

                        print('-' * 70)
                        print('Saving...')
                        Epoch = int((step * FLAGS.batch_size) / FLAGS.epoch_size)

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('training/', file)

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
    if tf.gfile.Exists('training/'):
        tf.gfile.DeleteRecursively('training/')
    tf.gfile.MakeDirs('training/')
    train()

if __name__ == '__main__':
    tf.app.run()