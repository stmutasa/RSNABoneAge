# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import os  # for the os type functionality, read write files and manipulate paths
import re  # regular expression operations for the print statements
import glob  # Simple but killer file reading module

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import Input

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/raw/', """Path to the data directory.""")

# Maybe define lambda for the regularalization penalty in the loss function ("weight decay" in tensorflow)
# Maybe define whether to use L1 or L2 regularization

# Global constants described in the input file to handle input sizes etc
IMAGE_SIZE = 0
NUM_CLASSES = 0
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0

# Constants we will use if we decide to use decaying learning rate
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average. (RMSProp)
NUM_EPOCHS_PER_DECAY = 2.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1.0  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.

TOWER_NAME = 'tower'  # If training on multiple GPU's, prefix all op names with tower_name


def forward_pass(images, phase_train1=True):
    """ This function builds the network architecture and performs the forward pass
        Args: Images = our input dictionary
        Returns: Logits (log odds units)
        Use tf.get_variable() in case we have multiple GPU's. get_variable creates or retreives a variable only in the
        scope defined by the block of code under variable_scope. This allows us to reuse variables in each block"""
    # normal kernel sizes: 96, 2048, 1024, 1024, 1024, 1024, 512

    # Set phase train to true if this is a training forward pass. Change the python bool to a tensorflow bool
    phase_train = tf.Variable(phase_train1, dtype=tf.bool, trainable=False)

    # The first convolutional layer
    with tf.variable_scope('conv1') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[7, 7, 1, 96], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')  # Create a 2D tensor with BATCH_SIZE rows

        # Insert batch norm layer:
        # norm = batch_norm_layer(conv, 96, 'norm1', phase_train) My version

        # TF Dev version
        # norm = batch_norm(conv, decay=0.999, center=True, scale=True, updates_collections=None,
        #                     is_training=True, reuse=None, trainable=True)

        # Contrib version
        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        conv1 = tf.nn.elu(norm, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv1)  # Create a histogram/scalar summary of the conv1 layer


    # The second convolutional layer
    with tf.variable_scope('conv2') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 96, 256], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows

        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        conv2 = tf.nn.elu(norm, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv2)  # Create a histogram/scalar summary of the conv1 layer

    # The third convolutional layer
    with tf.variable_scope('conv3') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows

        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        conv3 = tf.nn.elu(norm, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv3)  # Create a histogram/scalar summary of the conv1 layer


    # The 4th convolutional layer
    with tf.variable_scope('conv4') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(conv3, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows

        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        conv4 = tf.nn.elu(norm, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv4)  # Create a histogram/scalar summary of the conv1 layer


    # To Do: Insert the affine transform layer here
    # affine1 = 0

    # The 5th convolutional layer
    with tf.variable_scope('conv5') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(conv4, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows

        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        conv5 = tf.nn.elu(norm, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv5)  # Create a histogram/scalar summary of the conv1 layer

    # To Do: Maybe apply dropout here

    # The Fc7 layer
    with tf.variable_scope('linear1') as scope:
        reshape = tf.reshape(conv5, [FLAGS.batch_size, -1])  # Move everything to n by b matrix for a single matmul
        dim = reshape.get_shape()[1].value  # Get columns for the matrix multiplication
        weights = _variable_with_weight_decay('weights', shape=[dim, 128], wd=0.0)

        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(weights, activation_fn=tf.nn.relu, is_training=True,
                                                            reuse=None),
                       lambda: tf.contrib.layers.batch_norm(weights, activation_fn=tf.nn.relu,
                                                            is_training=False, reuse=True, scope='norm'))

        fc7 = tf.nn.elu(tf.matmul(reshape, norm), name=scope.name)  # returns mat of size batch x 512
        _activation_summary(fc7)

    # The linear layer
    with tf.variable_scope('linear2') as scope:
        W = tf.Variable(np.random.randn(128, 1), name='Weights', dtype=tf.float32)
        b = tf.Variable(np.ones(FLAGS.batch_size), name='Bias', dtype=tf.float32)
        Logits1 = tf.add(tf.matmul(fc7, W), b, name=scope.name)
        Logits = tf.slice(Logits1, [0, 0], [1, 4])
        _activation_summary(Logits)

    return Logits  # Return whatever the name of the final logits variable is


def batch_norm_layer(input_layer, kernels, scope, phase_train):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[kernels]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[kernels]), name='gamma', trainable=True)
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([mean, variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mean), tf.identity(variance)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(mean), ema.average(variance)))
        normed = tf.nn.batch_normalization(input_layer, mean, var, beta, gamma, 1e-3)

        return normed


def total_loss(logits, labels):
    """ Add L2 loss to the trainable variables and a summary
        Args:
            logits: logits from the forward pass
            labels the true input labels, a 1-D tensor with 1 value for each image in the batch
        Returns:
            Your loss value as a Tensor (float)
    """
    # Calculate MSE loss: square root of the mean of the square of an elementwise subtraction of logits and labels
    loss = tf.reduce_mean(tf.square(labels - logits))

    # Output the summary of the MSE
    tf.summary.scalar('Mean Square Error', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    # For now return MSE loss, add L2 regularization below later
    return loss

    # total_loss is cross entropy loss plus L2 loss. L2 loss is added to the collection "losses"
    #  when we use the _variable_with_weight decay function and a wd (lambda) value > 0
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')  # add_n is equal to a riemann sum operand


def backward_pass(total_loss, global_step1, lr_decay=False):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""
    if lr_decay:  # Use decaying learning rate if flag is true
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)  # how many steps until we decay
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step1, decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR, staircase=True)  # Can Change to another type
        tf.summary.scalar('learning_rate', lr)  # Output a scalar sumamry to TensorBoard

    # Compute the gradients. Control_dependencies waits until the operations in the parameter is executed before
    # executing the rest of the block. This makes sure we don't update gradients until we have calculated the backprop
    # with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(0.001)  # Create an AdamOptimizer graph: Can Change

    # Use the optimizer above to compute gradients to minimize total_loss.
    grads = opt.compute_gradients(total_loss)  # Returns a tensor with Gradients:Variable pairs

    # Now actually apply the gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step1)  # Returns an operation that applies the gradients

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    # These include the biases, the activation layers (nonlinearities) and weights
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Add histograms for the gradients we calculated above
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.summary.histogram(var.op.name + 'gradients', grad)

    # Per TF Documentation: "Certain training algorithms like momentum benefit from keeping track of the moving average of variables during
    # optimization. This improves results significantly."
    # Basically this is equivalent to RMS Prop, unsure if it is helpful when you use the Adam optimizer To Do
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step1)

    # Applies the RMS prop like normalization to all the variables in the trainable variables collection. Saves the
    # results to the moving average variables collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):  # Wait until we apply the gradients
        train_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return train_op


def _activation_summary(x):
    """ Helper to create summaries for activations
        Creates a summary to measure the proportion of your W in x that is all zero
        Parameters: x = a tensor
        Returns: Nothing"""
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)  # remove tower_[0-9] in multi GPU training
    tf.summary.histogram(tensor_name + '/activations', x)  # Output a summary protobuf with a histogram of x
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))  # " but with a scalar of the fraction of 0's

    return


def _variable_on_cpu(name: object, shape: object, initializer: object):
    """ Helper to create a variable stored on CPU memory.
        Why do this? Well if you are using multiple GPU's it might be faster to store some things on the CPU since
        copying from CPU to GPU is faster than GPU to GPU. Can change to let TF decide by deleting the cpu context code
        Args:
            name: The name of the variable
            shape: the list of ints
            initializer: the initializer for the vriable
        Returns:
            Variable tensor"""
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var


def _variable_with_weight_decay(name, shape, wd):
    """ Helper to initialize a normally distributed variable with weight decay if wd is specified.
    Args:
        name, the name
        shape, the list of ints
        stddev, the standard deviation of a truncated gussian
        wd: add L2 loss decay multiplied by this float Can Change to L1 weight decay if needed"""
    dtype = tf.float32

    # Use the Xavier initializer here. Can Change to truncated normal
    # var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(dtype=dtype))
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=5e-2, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')  # Uses half the L2 loss of Var*wd
        tf.add_to_collection('losses', weight_decay)  # Add the calculated weight decay to the collection of losses

    return var


def inputs(num_epochs):
    """ This function loads our raw inputs, processes them to a protobuffer that is then saved and
        loads the protobuffer into a batch of tensors """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not os.path.isfile('data/boneageproto.tfrecords') and not os.path.isfile('data/boneageloadict'):

        # Part 1: Load the raw images and labels dictionary ---------------------------
        print('----------------------No existing records -- Loading Raw Data...')
        images = {}

        # First load the raw data using the handy glob library
        globs = glob.glob(FLAGS.data_dir + '*.jpg')  # Returns a list of filenames

        i = 0
        for file_id in globs:  # Loop through every jpeg in the data directory
            raw = Input.read_image(file_id)  # First read the image into a unit8 numpy array named raw
            #raw = Input.pre_process_image(raw)  # Apply the pre processing of the image

            # Append the dictionary with the key: value pair of the basename (not full globname) and processed image
            images[os.path.splitext(os.path.basename(file_id))[0]] = raw
            i += 1
            if i % 300 == 0:
                print('     %i Images Loaded at %s, generating sample...' % (i, raw.shape))  # Just to update us
                raw2 = (raw - np.mean(raw)) / np.std(raw)
                plt.title(file_id)
                plt.imshow(raw2, cmap=plt.cm.gray)
                plt.show()

        label_dir = os.path.join(FLAGS.data_dir,
                                 'handdictionary')  # The labels dict is saved under handdictionary binary
        labels = Input.read_labels(label_dir)  # Add the dictionary of labels we have

        # Part 2: Save the images and labels to protobuf -------------------------------
        print('------------------------------------Saving images to records...')
        Input.img_protobuf(images, labels, 'bonageproto')

    else:
        print('-------------------------Previously saved records found! Loading...')

    # Part 3: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    data = Input.load_protobuf(None, 'bonageproto', True)

    # Part 4: Create randomized batches
    print('----------------------------------Creating and randomizing batches...')
    data = Input.randomize_batches(data, FLAGS.batch_size)

    tf.summary.image('post randomize batches img', data['image'], max_outputs=1)

    return data
