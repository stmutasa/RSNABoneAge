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
import scipy.misc as scipy

import tensorflow as tf
import Input
import spatial_transformer as st

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('data_dir', 'data/raw/', """Path to the data directory.""")


TOWER_NAME = 'tower'  # If training on multiple GPU's, prefix all op names with tower_name


def forward_pass(images, phase_train1=True, bts=0):
    """ This function builds the network architecture and performs the forward pass
        Args: Images = our input dictionary
        Returns: Logits (log odds units)
        Use tf.get_variable() in case we have multiple GPU's. get_variable creates or retreives a variable only in the
        scope defined by the block of code under variable_scope. This allows us to reuse variables in each block"""
    # normal kernel sizes: 96, 2048, 1024, 1024, 1024, 1024, 512

    # Adjust the batch size for training versus testing
    if bts:
        batch_size = bts
    else:
        batch_size = FLAGS.batch_size

    # Set Phase train variable
    phase_train = tf.Variable(phase_train1, trainable=False, dtype=tf.bool)

    # The first convolutional layer. Dimensions: 4, 128, 128, 64
    conv1 = convolution('Conv1', images, 7, 64, phase_train=phase_train)

    # The second convolutional layer    Dimensions: _, 64, 64, 128
    conv2 = convolution('Conv2', conv1, 5, 128, phase_train=phase_train)

    # Inception layer
    inception = inception_layer('Inception', conv2, 32, phase_train=phase_train)

    # The third convolutional layer Dimensions: _,32, 32, 256
    conv3 = convolution('Conv3', inception, 3, 256, phase_train=phase_train)

    # Insert inception/residual layer here. Output is same dimensions as previous layer
    residual1 = residual_layer('Residual', conv3, 3, 64, 'SAME', phase_train)

    # The 4th convolutional layer   Dimensions: _, 16, 16, 128
    conv4 = convolution('Conv4', residual1, 3, 128, phase_train=phase_train)

    # The affine transform layer here: Dimensions: _, 16, 16, 128
    with tf.variable_scope('Transformer') as scope:

        # Set up the localisation network to calculate floc(u):
        W1 = tf.get_variable('Weights1', shape=[16 * 16 * 128, 20],
                             initializer=tf.truncated_normal_initializer(stddev=5e-2))
        B1 = tf.get_variable('Bias1', shape=[20], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        W2 = tf.get_variable('Weights2', shape=[20, 6], initializer=tf.truncated_normal_initializer(stddev=5e-2))

        # Add weights to collection
        tf.add_to_collection('weights', W1)
        tf.add_to_collection('weights', W2)

        # Always start with the identity transformation
        initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        B2 = tf.Variable(initial_value=initial, name='Bias2')

        # Define the two layers of the localisation network
        H1 = tf.nn.tanh(tf.matmul(tf.zeros([batch_size, 16 * 16 * 128]), W1) + B1)
        H2 = tf.nn.tanh(tf.matmul(H1, W2) + B2)

        # Define the output size to the original dimensions
        output_size = (16, 16)
        h_trans = st.transformer(conv4, H2, output_size)

    # The 5th convolutional layer, Dimensions: _, 8, 8, 128
    conv5 = convolution('Conv5', h_trans, 1, 128, phase_train=phase_train)

    # The Fc7 layer Dimensions: _, 128
    with tf.variable_scope('linear1') as scope:
        reshape = tf.reshape(conv5, [batch_size, -1])  # [batch, ?]
        dim = reshape.get_shape()[1].value  # Get columns for the matrix multiplication
        weights = tf.get_variable('weights', shape=[dim, 128], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        tf.add_to_collection('weights', weights)
        fc7 = tf.nn.relu(tf.matmul(reshape, weights), name=scope.name)  # returns mat of size [batch x 128
        if phase_train1: fc7 = tf.nn.dropout(fc7, keep_prob=FLAGS.dropout_factor)  # Apply dropout here
        _activation_summary(fc7)

    # The linear layer Dimensions: 1x_
    with tf.variable_scope('linear2') as scope:
        W = tf.get_variable('Weights', shape=[128, 1], initializer=tf.truncated_normal_initializer(stddev=5e-2))
        tf.add_to_collection('weights', W)
        b = tf.Variable(np.ones(batch_size), name='Bias', dtype=tf.float32)
        Logits = tf.add(tf.matmul(fc7, W), b, name=scope.name)
        Logits = tf.slice(Logits, [0, 0], [batch_size, 1])
        Logits = tf.transpose(Logits)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)

    return Logits, L2_loss  # Return whatever the name of the final logits variable is


def convolution(scope, X, F, K, S=2, padding='SAME', phase_train=None):
    """
    This is a wrapper for convolutions
    :param scope:
    :param X: Output of the prior layer
    :param F: Convolutional filter size
    :param K: Number of feature maps
    :param S: Stride
    :param padding:
    :param phase_train: For batch norm implementation
    :return:
    """

    # Set channel size based on input depth
    C = X.get_shape().as_list()[3]

    # Set the scope
    with tf.variable_scope(scope) as scope:

        # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
        kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2))

        # Add to the weights collection
        tf.add_to_collection('weights', kernel)

        # Perform the actual convolution
        conv = tf.nn.conv2d(X, kernel, [1, S, S, 1], padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

        # Apply the batch normalization. Updates weights during training phase only
        norm = tf.cond(phase_train,
                lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                     updates_collections=None, is_training=True, reuse=None,
                                                     scope=scope, decay=0.9, epsilon=1e-5),
                lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                     updates_collections=None, is_training=False, reuse=True,
                                                     scope=scope, decay=0.9, epsilon=1e-5))



        # Relu activation
        conv = tf.nn.relu(norm, name=scope.name)

        # Create a histogram/scalar summary of the conv1 layer
        _activation_summary(conv)

        return conv


def inception_layer(scope, X, K, S=1, padding='SAME', phase_train=None):
    """
    This function implements an inception layer or "network within a network"
    :param scope:
    :param X: Output of the previous layer
    :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
    :param S: Stride
    :param padding:
    :param phase_train: For batch norm implementation
    :return: the inception layer output after concat
    """

    # Implement an inception layer here ----------------
    with tf.variable_scope(scope) as scope:

        # First branch, 1x1x64 convolution
        inception1 = convolution('Inception1', X, 1, K, S, phase_train=phase_train)  # 64x64x64

        # Second branch, 1x1 convolution then 3x3 convolution
        inception2a = convolution('Inception2a', X, 1, 1, 1, phase_train=phase_train)  # 64x64x1
        inception2 = convolution('Inception2', inception2a, 3, K, S, phase_train=phase_train)  # 64x64x64

        # Third branch, 1x1 convolution then 5x5 convolution:
        inception3a = convolution('Inception3a', X, 1, 1, 1, phase_train=phase_train)  # 64x64x1
        inception3 = convolution('Inception3', inception3a, 5, K, S, phase_train=phase_train)  # 64x64x64

        # Fourth branch, max pool then 1x1 conv:
        inception4a = tf.nn.max_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], padding)  # 64x64x256
        inception4 = convolution('Inception4', inception4a, 1, K, S, phase_train=phase_train)  # 64x64x64

        # Concatenate the results for dimension of 64,64,256
        inception = tf.concat([tf.concat([tf.concat([inception1, inception2], axis=3),
                                          inception3], axis=3), inception4], axis=3)

        return inception


def residual_layer(scope, X, F, K, padding='SAME', phase_train=None):
    """
    This is a wrapper for implementing a hybrid residual layer with inception layer as F(x)
    :param scope:
    :param X: Output of the previous layer
    :param F: Dimensions of the second convolution in F(x) - the non inception layer one
    :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
    :param S: Stride
    :param padding:
    :param phase_train: For batch norm implementation
    :return:
    """

    # Set channel size based on input depth
    C = X.get_shape().as_list()[3]

    # Set the scope. Implement a residual layer below: Conv-relu-conv-residual-relu
    with tf.variable_scope(scope) as scope:

        # The first layer is an inception layer
        conv1 = inception_layer(scope, X, K, 1, phase_train=phase_train)

        # Define the Kernel for conv2. Which is a normal conv layer
        kernel = tf.get_variable('Weights', shape=[F, F, C, K*4],
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2))

        # Add this kernel to the weights collection for L2 reg
        tf.add_to_collection('weights', kernel)

        # Perform the actual convolution
        conv2 = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

        # Add in the residual here
        residual = tf.add(conv2, X)

        # Apply the batch normalization. Updates weights during training phase only
        norm = tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope, decay=0.9, epsilon=1e-5))

        # Relu activation
        conv = tf.nn.relu(norm, name=scope.name)

        # Create a histogram/scalar summary of the conv1 layer
        _activation_summary(conv)

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


def backward_pass(total_loss):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Use learning rate decay
    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.lr_steps, FLAGS.lr_decay, staircase=True)
    tf.summary.scalar('learning_rate', lr)  # Output a scalar sumamry to TensorBoard

    # Compute the gradients. Multiple alternate methods grayed out. So far Adam is winning by a mile
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
    # opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum, use_nesterov=FLAGS.use_nesterov)
    # opt = tf.train.ProximalGradientDescentOptimizer(lr,l2_regularization_strength=FLAGS.l2_gamma)
    # opt = tf.train.GradientDescentOptimizer(lr)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # clip the gradients
    clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(clipped_gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    # These include the biases, the activation layers (nonlinearities) and weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

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


def inputs(skip=False):
    """ This function loads our raw inputs, processes them to a protobuffer that is then saved and
        loads the protobuffer into a batch of tensors """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip:

        # Part 1: Load the raw images and labels dictionary ---------------------------
        print('----------------------No existing records -- Loading Raw Data...')
        images = {}

        # First load the raw data using the handy glob library
        globs = glob.glob(FLAGS.data_dir + '*.jpg')  # Returns a list of filenames

        i = 0
        for file_id in globs:  # Loop through every jpeg in the data directory

            # First read the image into a unit8 numpy array named raw
            raw = Input.read_image(file_id)

            # convert to float32
            raw = scipy.imresize(raw, (512, 512))
            raw = raw.astype(np.int16)

            # Append the dictionary with the key: value pair of the basename (not full globname) and processed image
            images[os.path.splitext(os.path.basename(file_id))[0]] = raw
            i += 1
            if i % 100 == 0:
                print('     %i Images Loaded at %s, generating sample...' % (i, raw.shape))  # Just to update us

                # if i > 500: break FOR TESTING OVERFIT

        label_dir = os.path.join(FLAGS.data_dir, 'handdictionary')  # The labels dict
        labels = Input.read_labels(label_dir)  # Add the dictionary of labels we have

        # Part 2: Save the images and labels to protobuf -------------------------------
        print('------%s Images successfully loaded------------------Saving images to records...' % i)
        _ = Input.img_protobuf(images, labels, 'bonageproto')

    else:
        print('-------------------------Previously saved records found! Loading...')

    # Part 3: Load the protobuff  -----------------------------
    print('----------------------------------------Loading Protobuff...')
    data = Input.load_protobuf('bonageproto', True)
    validation = Input.load_validation_set('bonageproto')

    # Part 4: Create randomized batches
    print('----------------------------------Creating and randomizing batches...')
    data = Input.randomize_batches(data, FLAGS.batch_size)
    validation = Input.val_batches(validation, FLAGS.batch_size)

    return data, validation


def RunningMean(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def calculate_errors(predictions, label):
    """
    This function retreives the labels and predictions and then outputs the accuracy based on the actual
    standard deviations from the atlas of bone ages. The prediction is considered "right" if it's within
    two standard deviations
    :param predictions:
    :param labels:
    :param girls: Whether we're using the female or male standard deviations
    :return: Accurace : calculated as % of right/total
    """

    # First define our variables:
    right = 0.0  # Number of correct predictions
    total = predictions.size  # Number of total predictions
    std_dev = np.zeros_like(predictions, dtype='float32')  # The array that will hold our STD Deviations
    tot_err = 0.0

    # No apply the standard deviations
    for i in range(0, total):

        # Bunch of if statements assigning the STD for the patient's true age
        if FLAGS.model < 3: # Girls
            if label[i] <= (3 / 12): std_dev[i] = 0.72 / 12
            elif label[i] <= (6 / 12): std_dev[i] = 1.16 / 12
            elif label[i] <= (9 / 12): std_dev[i] = 1.36 / 12
            elif label[i] <= (12 / 12): std_dev[i] = 1.77 / 12
            elif label[i] <= (18 / 12): std_dev[i] = 3.49 / 12
            elif label[i] <= (24 / 12): std_dev[i] = 4.64 / 12
            elif label[i] <= (30 / 12): std_dev[i] = 5.37 / 12
            elif label[i] <= 3: std_dev[i] = 5.97 / 12
            elif label[i] <= 3.5: std_dev[i] = 7.48 / 12
            elif label[i] <= 4: std_dev[i] = 8.98 / 12
            elif label[i] <= 4.5: std_dev[i] = 10.73 / 12
            elif label[i] <= 5: std_dev[i] = 11.65 / 12
            elif label[i] <= 6: std_dev[i] = 10.23 / 12
            elif label[i] <= 7: std_dev[i] = 9.64 / 12
            elif label[i] <= 8: std_dev[i] = 10.23 / 12
            elif label[i] <= 9: std_dev[i] = 10.74 / 12
            elif label[i] <= 10: std_dev[i] = 11.73 / 12
            elif label[i] <= 11: std_dev[i] = 11.94 / 12
            elif label[i] <= 12: std_dev[i] = 10.24 / 12
            elif label[i] <= 13: std_dev[i] = 10.67 / 12
            elif label[i] <= 14: std_dev[i] = 11.3 / 12
            elif label[i] <= 15: std_dev[i] = 9.23 / 12
            else: std_dev[i] = 7.31 / 12

        else:   # Boys
            if label[i] <= (3 / 12): std_dev[i] = 0.72 / 12
            elif label[i] <= (6 / 12): std_dev[i] = 1.13 / 12
            elif label[i] <= (9 / 12): std_dev[i] = 1.43 / 12
            elif label[i] <= (12 / 12): std_dev[i] = 1.97 / 12
            elif label[i] <= (18 / 12): std_dev[i] = 3.52 / 12
            elif label[i] <= (24 / 12): std_dev[i] = 3.92 / 12
            elif label[i] <= (30 / 12): std_dev[i] = 4.52 / 12
            elif label[i] <= 3: std_dev[i] = 5.08 / 12
            elif label[i] <= 3.5: std_dev[i] = 5.40 / 12
            elif label[i] <= 4: std_dev[i] = 6.66 / 12
            elif label[i] <= 4.5: std_dev[i] = 8.36 / 12
            elif label[i] <= 5: std_dev[i] = 8.79 / 12
            elif label[i] <= 6: std_dev[i] = 9.17 / 12
            elif label[i] <= 7: std_dev[i] = 8.91 / 12
            elif label[i] <= 8: std_dev[i] = 9.10 / 12
            elif label[i] <= 9: std_dev[i] = 9.0 / 12
            elif label[i] <= 10: std_dev[i] = 9.79 / 12
            elif label[i] <= 11: std_dev[i] = 10.09 / 12
            elif label[i] <= 12: std_dev[i] = 10.38 / 12
            elif label[i] <= 13: std_dev[i] = 10.44 / 12
            elif label[i] <= 14: std_dev[i] = 10.72 / 12
            elif label[i] <= 15: std_dev[i] = 11.32 / 12
            elif label[i] <= 16: std_dev[i] = 12.86 / 12
            else: std_dev[i] = 13.05 / 12

        # Calculate the MAE
        if predictions[i] < 0: predictions[i] = 0
        if predictions[i] > 18: predictions[i] = 18
        abs_err = abs(predictions[i] - label[i])
        tot_err += abs_err

        # Mark it right if we are within 2 std_devs
        if abs_err <= (std_dev[i] * 2):  # If difference is less than 2 stddev
            right += 1

    accuracy = (right / total) * 100  # Calculate the percent correct
    mae = (tot_err / total)

    return accuracy, mae


def after_run(predictions1, label1, loss1, loss_value, step, duration):
    # First print the number of examples per step
    eg_s = FLAGS.batch_size / duration
    print('Step %d, L2 Loss: = %.8f (%.1f eg/s;)' % (step, loss_value, eg_s), end=" ")

    predictions = predictions1.astype(np.float)
    label = label1.astype(np.float)

    # Calculate the accuracy
    acc, mae = calculate_errors(predictions, label)

    # Print the summary
    np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
    print('Eg. Predictions: Network(Real): %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), %.1f (%.1f), '
          'MSE: %.4f, MAE: %.2f Yrs, Train Accuracy: %s %%'
          % (predictions[0], label[0], predictions[1], label[1], predictions[2],
             label[2], predictions[3], label[3], loss1, mae, acc))


def imgshow(nda, plot=True, title=None, margin=0.05):
    """ Helper function to display a numpy array using matplotlib
    Args:
        nda: The source image as a numpy array
        title: what to title the picture drawn
        margin: how wide a margin to use
        plot: plot or not
    Returns:
        none"""

    # Set up the figure object
    fig = plt.figure()
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    # The rest is standard matplotlib fare
    plt.set_cmap("gray")  # Print in greyscale
    ax.imshow(nda)

    if title: plt.title(title)
    if plot: plt.show()