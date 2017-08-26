"""
This folder is for the RSNA bone age competition
To keep it separate.

Train
"""

import os, time, re

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
tf.app.flags.DEFINE_integer('num_gpus', 3, """number of GPU's to use""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_integer('num_epochs', 900, """Number of epochs to run""")
tf.app.flags.DEFINE_string('gender', 'F', """Which version to run""")

# Female = 5958, 950 @ 64, Male = 6934, 108 @ 64, YF: 3036, 47, OF: 3458, 54
tf.app.flags.DEFINE_integer('epoch_size', 5958, """How many images were loaded""")
tf.app.flags.DEFINE_integer('print_interval', 93, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_steps', 940, """How many STEPS to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma',1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay', 0.98, """The base factor for exp learning rate decay""")
tf.app.flags.DEFINE_integer('lr_steps', 4000, """ The number of steps until we decay the learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('loss_factor', 0.0, """Addnl. fac. for the cost sensitive loss (2 makes 0 == 3x more)""")

def tower_loss(scope, images, labels):
    """
    Runs a forward pass and total loss step for one tower
    :param scope: To identify the tower
    :param images: input images
    :param labels: input labels
    :return:  total loss tensor
    """

    # First forward pass
    logits, _ = Competition.forward_pass_res(images, phase_train1 = True)

    # Normalize labels
    labels = tf.divide(labels, 19)

    # now the loss for this tower
    _ = Competition.total_loss(logits, labels)

    # Assemble losses for just this tower
    losses = tf.get_collection('losses', scope)

    # Add the losses for this tower
    #total_loss = tf.add(MSE_loss, l2loss, name='TotalLoss')
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('Tower_[0-9]*/', '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss, tf.multiply(logits, 19)


def average_gradients(tower_grads):
    """
    Calculates the average gradient for each shared variable across all towers
    This function is the synchronization point
    :param tower_grads: Tuple list of the gradients (gradient, variable)
                        the outer list is over individual gradients the inner list is per tower
    :return: list of tuples where the gradient has been averaged across all towers
    """

    average_grads = []

    # loop the outer list
    for grad_and_vars in zip(*tower_grads):

        # each grad_vars looks like: ((grad0_gpu0, var0_gpu_0) ... (grad0_gpuN, var0_gpu_N))
        grads = []

        # loop the inenr list
        for g, _ in grad_and_vars:

            # Add 0 dimension to the gradients to represent the tower
            expanded_g = tf.expand_dims(g, 0)

            # Append a "tower" dimension to everage over below
            grads.append(expanded_g)

        # Average over the 'tower' dimension
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Return the first tower's pointer to the variable since variables are redundant
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

        return average_grads


def backward_pass_multiGPU(data_queue):

    """
    Does the backward pass for the multi GPU training
    :param images:
    :param labels:
    :return: train op - final op to train
    """

    # Define global step
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # Create an optimizer
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=1e-8)

    # Calculate the gradients for each tower
    tower_grads = []

    # Iterate over all the gpus and set unique scope name for each tower
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % i) as scope:

                    # Dequeue one batch for the GPU
                    images, labels = data_queue.dequeue()

                    # Calculate the loss for this tower
                    loss, logits = tower_loss(scope, images, labels)

                    # Reuse the variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this tower
                    grads = opt.compute_gradients(loss)

                    # Append the gradients to list for all towers
                    tower_grads.append(grads)

    # Now calculate the average gradients, (Sync point)
    grads = average_gradients(tower_grads)

    # Add histograms for gradients
    for grad, var in grads:
        if grad is not None: summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates into a single train op
    train_op = tf.group(apply_gradient_op, variable_averages_op)

    return train_op, summaries, logits


def train():

    # First, make this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Load the images and labels.
        data, _ = Competition.Inputs(skip=True)

        # batch queue method
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([data['image'], data['reading']],
                                                                    capacity=2 * FLAGS.num_gpus)

        # Multi GPU forward and backward pass
        #train_op, summaries, logits = backward_pass_multiGPU(data['image'], data['reading'])
        train_op, summaries, logits = backward_pass_multiGPU(batch_queue)

        # For sess.run
        labels2 = data['reading']

        # -------------------  Housekeeping functions  ----------------------

        # Merge the summaries
        #all_summaries = tf.summary.merge_all()
        all_summaries = tf.summary.merge(summaries)

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=2)

        # TODO
        all_summaries = tf.summary.merge(summaries)

        # Initialize variables operation
        #var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        var_init = tf.global_variables_initializer()

        # -------------------  Session Initializer  ----------------------

        # config Proto sets options for configuring the session like run on GPU, allocate GPU memory etc.
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as mon_sess:

            # Initialize the variables
            mon_sess.run(var_init)

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter('training/Log1/', mon_sess.graph)

            # Initialize the thread coordinator
            #coord = tf.train.Coordinator()

            # Start the queue runners
            #threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)
            threads = tf.train.start_queue_runners(sess=mon_sess)

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

                        # Also retreive the predictions and labels
                        lbl, preds = mon_sess.run([labels2, logits])

                        # Convert to numpy arrays
                        predictions = np.squeeze(preds.astype(np.float))
                        label = np.squeeze(lbl.astype(np.float))

                        # Clip predictions
                        predictions[predictions<0] = 0
                        predictions[predictions>19] = 19

                        # Calculate MAE
                        MAE = np.mean(np.absolute((predictions - label)))

                        # Now print the loss values
                        print ('-'*70)
                        print('Step: %s, Time Elapsed: %.1f sec, %s examples' % (step, duration, len(label)))

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
                        file = ('T1Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('training/', file)

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                    # Increment step
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done with Training - Epoch limit reached')

            finally:

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