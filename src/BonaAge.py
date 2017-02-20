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
# import sys  # access to variables used by the interpreter (which reads and executes python code
import glob  # Simple but killer file reading module

import tensorflow as tf
import Input

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('batch_size', 2, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/raw/', """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_float('keep_prob', 0.5, """probability of dropping out a neuron""")
tf.app.flags.DEFINE_integer('num_examples', 1384, """The amount of source images""")
tf.app.flags.DEFINE_integer('num_classes', 40, """ The number of classes """)

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


def forward_pass(images):
    """ This function builds the network architecture and performs the forward pass
        Args: Images = our input dictionary
        Returns: Logits (log odds units)
        Use tf.get_variable() in case we have multiple GPU's. get_variable creates or retreives a variable only in the
        scope defined by the block of code under variable_scope. This allows us to reuse variables in each block"""
    # normal kernel sizes: 96, 2048, 1024, 1024, 1024, 1024, 512

    # The first convolutional layer
    with tf.variable_scope('conv1') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[7, 7, 1, 96], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')  # Create a 2D tensor with BATCH_SIZE rows
        biases = _variable_on_cpu('biases', 96, tf.constant_initializer(0.0))  # Initialize biases as 0
        pre_activation = tf.nn.bias_add(conv, biases)  # Add conv and biases into one tensor
        conv1 = tf.nn.elu(pre_activation, name=scope.name)  # Use ELU to prevent sparsity.
        #   _activation_summary(conv1)  # Create a histogram/scalar summary of the conv1 layer

    # Insert batch norm layer:
    norm1 = tf.nn.lrn(conv1, 4, 1.0, 0.001 / 9.0, 0.75, 'norm1')

    # The second convolutional layer
    with tf.variable_scope('conv2') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 96, 256], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows
        biases = _variable_on_cpu('biases', 256, tf.constant_initializer(0.0))  # Initialize biases as 0
        pre_activation = tf.nn.bias_add(conv, biases)  # Add conv and biases into one layer
        conv2 = tf.nn.elu(pre_activation, name=scope.name)  # Use ELU to prevent sparsity.
        #   _activation_summary(conv2)  # Create a histogram/scalar summary of the conv1 layer

    # Insert batch norm layer:
    norm2 = tf.nn.lrn(conv2, 4, 1.0, 0.001 / 9.0, 0.75, 'norm2')

    # The third convolutional layer
    with tf.variable_scope('conv3') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(norm2, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows
        biases = _variable_on_cpu('biases', 128, tf.constant_initializer(0.0))  # Initialize biases as 0
        pre_activation = tf.nn.bias_add(conv, biases)  # Add conv and biases into one layer
        conv3 = tf.nn.elu(pre_activation, name=scope.name)  # Use ELU to prevent sparsity.
        #   _activation_summary(conv3)  # Create a histogram/scalar summary of the conv1 layer

    # Insert batch norm layer:
    norm3 = tf.nn.lrn(conv3, 4, 1.0, 0.001 / 9.0, 0.75, 'norm3')

    # The 4th convolutional layer
    with tf.variable_scope('conv4') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(norm3, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows
        biases = _variable_on_cpu('biases', 128, tf.constant_initializer(0.0))  # Initialize biases as 0
        pre_activation = tf.nn.bias_add(conv, biases)  # Add conv and biases into one layer
        conv4 = tf.nn.elu(pre_activation, name=scope.name)  # Use ELU to prevent sparsity.
        #   _activation_summary(conv4)  # Create a histogram/scalar summary of the conv1 layer

    # Insert batch norm layer:
    norm4 = tf.nn.lrn(conv4, 4, 1.0, 0.001 / 9.0, 0.75, 'norm4')

    # To Do: Insert the affine transform layer here
    affine1 = 0

    # The 5th convolutional layer
    with tf.variable_scope('conv5') as scope:  # Define this variable scope as conv1
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.0)  # Xavier init. No WD
        conv = tf.nn.conv2d(norm4, kernel, [1, 2, 2, 1], padding='VALID')  # Create a 2D tensor with BATCH_SIZE rows
        biases = _variable_on_cpu('biases', 128, tf.constant_initializer(0.0))  # Initialize biases as 0
        pre_activation = tf.nn.bias_add(conv, biases)  # Add conv and biases into one layer
        conv5 = tf.nn.elu(pre_activation, name=scope.name)  # Use ELU to prevent sparsity.
        #   _activation_summary(conv5)  # Create a histogram/scalar summary of the conv1 layer

    # The last batch norm layer:
    norm5 = tf.nn.lrn(conv5, 4, 1.0, 0.001 / 9.0, 0.75, 'norm5')

    # To Do: Maybe apply dropout here

    # The Fc7 layer
    with tf.variable_scope('linear1') as scope:
        reshape = tf.reshape(norm5, [FLAGS.batch_size, -1])  # Move everything to n by b matrix for a single matmul
        dim = reshape.get_shape()[1].value  # Get columns for the matrix multiplication
        weights = _variable_with_weight_decay('weights', shape=[dim, 128], wd=0.0)
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        fc7 = tf.nn.elu(tf.matmul(reshape, weights) + biases, name=scope.name)  # returns mat of size batch x 512
        #   _activation_summary(fc7)

    # The Fc8 layer
    with tf.variable_scope('linear2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[128, 64], wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        fc8 = tf.nn.elu(tf.matmul(fc7, weights) + biases, name=scope.name)
        #   _activation_summary(fc8)

    # The linear layer
    with tf.variable_scope('softmax') as scope:
        weights = _variable_with_weight_decay('weights', shape=[64, FLAGS.num_classes], wd=0.0)
        biases = _variable_on_cpu('biases', [FLAGS.num_classes], tf.constant_initializer(0.0))
        softmax = tf.add(tf.matmul(fc8, weights), biases, name=scope.name)
        #   _activation_summary(softmax)

    return softmax  # Return whatever the name of the final logits variable is


def total_loss(logits, labels):
    """ Add L2 loss to the trainable variables and a summary
        Args:
            logits: logits from the forward pass
            labels the true input labels, a 1-D tensor with 1 value for each image in the batch
        Returns:
            Your loss value as a Tensor (float)"""
    # labels = tf.cast(labels, tf.float32)  # Changes the labels input to a 64 bit integer
    # use the sparse version of cross entropy when each class true label is exclusive (1 for correct class, 0 ee)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')

    # The function above returns a matrix, the one below computes the mean of the values in the returned matrix
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    # Add a comparison of the cross entropy mean to the labels here and add to collection

    # Adds cross entropy mean to the graph collection 'losses'. Collections are basically persistent variables you can
    # retreive later at any time
    tf.add_to_collection('losses', cross_entropy_mean)

    # total_loss is cross entropy loss plus L2 loss. L2 loss is added to the collection "losses"
    #  when we use the _variable_with_weight decay function and a wd (lambda) value > 0
    return tf.add_n(tf.get_collection('losses'), name='total_loss')  # add_n is equal to a riemann sum operand


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

    # Generate moving averages of all losses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)  # To do

    # Compute the gradients. Control_dependencies waits until the operations in the parameter is executed before
    # executing the rest of the block. This makes sure we don't update gradients until we have calculated the backprop
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()  # Create an AdamOptimizer graph: Can Change

        # Use the optimizer above to compute gradients to minimize total_loss.
        grads = opt.compute_gradients(total_loss)  # Returns a tensor with Gradients:Variable pairs

    # Now actually apply the gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step1)  # Returns an operation that applies the gradients

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    # These include the biases, the activation layers (nonlinearities) and weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

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


def _variable_on_cpu(name: object, shape: object, initializer: object) -> object:
    """ Helper to create a variable stored on CPU memory.
        Why do this? Well if you are using multiple GPU's it might be faster to store some things on the CPU since
        copying from CPU to GPU is faster than GPU to GPU. Can change to let TF decide by deleting the cpu context code
        Args:
            name: The name of the variable
            shape: the list of ints
            initializer: the initializer for the vriable
        Returns:
            Variable tensor"""
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var


def _variable_with_weight_decay(name, shape, wd):
    """ Helper to initialize a normally distributed variable with weight decay if wd is specified.
    Args:
        name, the name
        shape, the list of ints
        stddev, the standard deviation of a truncated gussian
        wd: add L2 loss decay multiplied by this float Can Change to L1 weight decay if needed"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    # Use the Xavier initializer here. Can Change
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')  # Uses half the L2 loss of Var*wd
        tf.add_to_collection('losses', weight_decay)  # Add the calculated weight decay to the collection of losses

    return var


def _add_loss_summaries(total_loss):
    """ Generates the moving average for all losses and associated scalar summaries
            Args:
                Total loss: the loss calculated in the total_loss function
            Returns:
                loss_averages_op: an op for generating the moving average of losses"""

    # Compute the moving average of all individual losses and the total loss
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')  # Retreive the losses variable collection

    # creates shadow variables and ops to maintain moving averages
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # attach a scalar summary to all individual losses and the total loss and average losses
    for l in losses + [total_loss]:
        # Original loss is named with 'raw' for tensorboard, the moving average loss is just the name
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def inputs(num_epochs):
    """ This function loads our raw inputs, processes them to a protobuffer that is then saved and
        loads the protobuffer into a batch of tensors """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    #if not os.path.isfile('data/boneageproto.tfrecords') and not os.path.isfile('data/boneageloadict'):

    # Part 1: Load the raw images and labels dictionary ---------------------------
    # print('----------------------No existing records -- Loading Raw Data...')
    images = {}

    # First load the raw data using the handy glob library
    globs = glob.glob(FLAGS.data_dir + '*.jpg')  # Returns a list of filenames
    print(globs)
    i = 0
    for file_id in globs:  # Loop through every jpeg in the data directory
        raw = Input.read_image(file_id)  # First read the image into a unit8 numpy array named raw
        raw = Input.pre_process_image(raw)  # Apply the pre processing of the image

        # Append the dictionary with the key: value pair of the basename (not full globname) and processed image
        images[os.path.splitext(os.path.basename(file_id))[0]] = raw
        i += 1
        if i % 100 == 0: print('     %i Images Loaded %s' % (i, raw.shape))  # Just to update us

    label_dir = os.path.join(FLAGS.data_dir, 'handdictionary')  # The labels dict is saved under handdictionary binary
    labels = Input.read_labels(label_dir)  # Add the dictionary of labels we have

    # Part 2: Save the images and labels to protobuf -------------------------------
    print('------------------------------------Saving images to records...')
    Input.img_protobuf(images, labels, 'bonageproto')

    # else:
    #print('-------------------------Previously saved records found! Loading...')

    # Part 3: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    data = Input.load_protobuf(None, 'bonageproto', True)

    # Part 4: Create randomized batches
    print('----------------------------------Creating and randomizing batches...')
    data = Input.randomize_batches(data, FLAGS.batch_size)

    return data
