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

import tensorflow as tf
import Input

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, """Initial learning rate""")
tf.app.flags.DEFINE_string('data_dir', 'data/raw/', """Path to the data directory.""")

# Constants we will use if we decide to use decaying learning rate
MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average. (RMSProp)
NUM_EPOCHS_PER_DECAY = 2.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1.0  # Learning rate decay factor.
Num_Examples = 16

TOWER_NAME = 'tower'  # If training on multiple GPU's, prefix all op names with tower_name


def forward_pass(images, keep_prob=1.0, phase_train1=True):
    """ This function builds the network architecture and performs the forward pass
        Args: Images = our input dictionary
        Returns: Logits (log odds units)
        Use tf.get_variable() in case we have multiple GPU's. get_variable creates or retreives a variable only in the
        scope defined by the block of code under variable_scope. This allows us to reuse variables in each block"""
    # normal kernel sizes: 96, 2048, 1024, 1024, 1024, 1024, 512

    # Set phase train to true if this is a training forward pass. Change the python bool to a tensorflow bool
    phase_train = tf.Variable(phase_train1, dtype=tf.bool, trainable=False)

    # The first convolutional layer
    conv1 = convolution('Conv1', images, 1, 7, 96, phase_train=phase_train)

    # The second convolutional layer
    conv2 = convolution('Conv2', conv1, 96, 5, 256, phase_train=phase_train)

    # The third convolutional layer
    conv3 = convolution('Conv3', conv2, 256, 3, 128, phase_train=phase_train)

    # The 4th convolutional layer
    conv4 = convolution('Conv4', conv3, 128, 3, 128, phase_train=phase_train)

    # # To Do: Insert the affine transform layer here: Output of conv4 is [batch, 14,14,128]
    # with tf.variable_scope('Transformer') as scope:
    #
    #     # Set up the localisation network to calculate floc(u):
    #     W1 = tf.get_variable('Weights1', shape=[FLAGS.batch_size, 14, 128, 20], initializer=tf.contrib.layers.xavier_initializer())
    #     B1 = tf.get_variable('Bias1', shape=[20], initializer=tf.contrib.layers.xavier_initializer())
    #     W2 = tf.get_variable('Weights2', shape=[FLAGS.batch_size, 14, 20, 6], initializer=tf.contrib.layers.xavier_initializer())
    #
    #     # Always start with the identity transformation
    #     initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    #     initial = initial.astype('float32')
    #     initial = initial.flatten()
    #     B2 = tf.Variable(initial_value=initial, name='IDFunction')
    #
    #     # Define the two layers of the localisation network
    #     H1 = tf.nn.tanh(tf.matmul(conv4, W1) + B1)
    #     H2 = tf.nn.tanh(tf.matmul(H1, W2) + B2)
    #
    #     # Define the output size to the original dimensions
    #     output_size = (FLAGS.batch_size, 14, 14, 1)
    #     trans = transformer([-1, 14, 14, 128], H2, output_size)

    # The 5th convolutional layer
    # conv5 = convolution('Conv5', trans, 1, 3, 128, phase_train=phase_train)
    conv5 = convolution('Conv5', conv4, 128, 3, 128, phase_train=phase_train)

    # The Fc7 layer
    with tf.variable_scope('linear1') as scope:
        reshape = tf.reshape(conv5, [FLAGS.batch_size, -1])  # Move everything to n by b matrix for a single matmul
        dim = reshape.get_shape()[1].value  # Get columns for the matrix multiplication
        weights = tf.get_variable('weights', shape=[dim, 128], initializer=tf.contrib.layers.xavier_initializer())
        fc7 = tf.nn.relu(tf.matmul(reshape, weights), name=scope.name)  # returns mat of size batch x 512
        fc7 = tf.nn.dropout(fc7, keep_prob=keep_prob)  # Apply dropout here
        _activation_summary(fc7)

    # The linear layer
    with tf.variable_scope('linear2') as scope:
        W = tf.Variable(np.random.randn(128, 1), name='Weights', dtype=tf.float32)
        b = tf.Variable(np.ones(FLAGS.batch_size), name='Bias', dtype=tf.float32)
        Logits = tf.add(tf.matmul(fc7, W), b, name=scope.name)
        Logits = tf.slice(Logits, [0, 0], [FLAGS.batch_size, 1])
        Logits = tf.transpose(Logits)

    # Calculate the L2 regularization penalty
    L2_loss = FLAGS.l2_gamma * (tf.nn.l2_loss(conv1) + tf.nn.l2_loss(conv2) + tf.nn.l2_loss(conv3) +
                                tf.nn.l2_loss(conv4) + tf.nn.l2_loss(conv5) + tf.nn.l2_loss(fc7) + tf.nn.l2_loss(W))

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Create a summary scalar of L2 loss
    tf.summary.scalar('L2 Loss Penalty', L2_loss)

    return Logits, L2_loss  # Return whatever the name of the final logits variable is


def convolution(scope, X, C, F, K, S=2, padding='VALID', phase_train=None):
    with tf.variable_scope(scope) as scope:
        kernel = tf.get_variable('Weights', shape=[F, F, C, K], initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(X, kernel, [1, S, S, 1], padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

        # norm = tf.cond(phase_train,
        #                lambda: tf.contrib.layers.batch_norm(conv, decay=0.999, is_training=True, reuse=None),
        #                lambda: tf.contrib.layers.batch_norm(conv, is_training=False, reuse=True, scope='norm'))

        conv = tf.nn.relu(conv, name=scope.name)  # Use ELU to prevent sparsity.
        _activation_summary(conv)  # Create a histogram/scalar summary of the conv1 layer
        return conv


def total_loss(logits, labels):
    """ Add Lloss to the trainable variables and a summary
        Args:
            logits: logits from the forward pass
            labels the true input labels, a 1-D tensor with 1 value for each image in the batch
        Returns:
            Your loss value as a Tensor (float)
    """
    # Calculate MSE loss: square root of the mean of the square of an elementwise subtraction of logits and labels
    MSE_loss = tf.reduce_mean(tf.square(labels - logits))

    # Output the summary of the MSE and MAE
    tf.summary.scalar('Mean Square Error', MSE_loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', MSE_loss)

    # For now return MSE loss, add L2 regularization below later
    return MSE_loss


def backward_pass(total_loss, global_step1):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Compute the gradients. Multiple alternate methods grayed out. So far Adam is winning by a mile
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999)
    # opt = tf.train.ProximalGradientDescentOptimizer(lr,l2_regularization_strength=FLAGS.l2_gamma)
    # opt = tf.train.GradientDescentOptimizer(lr)

    # Compute then apply the gradients
    train_op = opt.minimize(total_loss, global_step1, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    # These include the biases, the activation layers (nonlinearities) and weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

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


def inputs():
    """ This function loads our raw inputs, processes them to a protobuffer that is then saved and
        loads the protobuffer into a batch of tensors """

    # First request what gender to train:
    gender = input('Please enter what Gender you would like to train: -> ')
    age = input('Now enter what age group you would like to train: -> ')

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not os.path.isfile('data/boneageproto.tfrecords'):  # and not os.path.isfile('data/boneageloadict'):

        # Part 1: Load the raw images and labels dictionary ---------------------------
        print('----------------------No existing records -- Loading Raw Data...')
        images = {}

        # First load the raw data using the handy glob library
        globs = glob.glob(FLAGS.data_dir + '*.jpg')  # Returns a list of filenames

        i = 0
        for file_id in globs:  # Loop through every jpeg in the data directory

            raw = Input.read_image(file_id)  # First read the image into a unit8 numpy array named raw

            # Append the dictionary with the key: value pair of the basename (not full globname) and processed image
            images[os.path.splitext(os.path.basename(file_id))[0]] = raw
            i += 1
            if i % 100 == 0:
                print('     %i Images Loaded at %s, generating sample...' % (i, raw.shape))  # Just to update us

            if i > 500: break

        label_dir = os.path.join(FLAGS.data_dir, 'handdictionary')  # The labels dict
        labels = Input.read_labels(label_dir)  # Add the dictionary of labels we have

        # Part 2: Save the images and labels to protobuf -------------------------------
        print('------------------------------------Saving images to records...')
        Input.img_protobuf(images, labels, 'bonageproto', gender=gender.upper(), age=int(age))

    else:
        print('-------------------------Previously saved records found! Loading...')

    # Part 3: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    data = Input.load_protobuf(None, 'bonageproto', True)

    # Part 4: Create randomized batches
    print('----------------------------------Creating and randomizing batches...')
    data = Input.randomize_batches(data, FLAGS.batch_size)

    return data
