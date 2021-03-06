"""
This folder is for the RSNA bone age competition
To keep it separate.

Functions:
Pre process
Protobuf
Input
Forward Pass
Loss
Backward pass
Train
"""

import os

import SODLoader as SDL
import SODNetwork as SDN
import numpy as np
import tensorflow as tf

# Define loader instances
sdl = SDL.SODLoader(os.getcwd())
sdn = SDN.SODMatrix()

# Define flags
FLAGS = tf.app.flags.FLAGS


def forward_pass_res(images, phase_train1=True):
    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    # Set Phase train variable
    phase_train = tf.Variable(phase_train1, trainable=False, dtype=tf.bool)

    # The first layer.
    conv1 = sdn.convolution('Conv1', images, 5, 32, 1, phase_train=phase_train, BN=False, relu=False)

    # The second  layer
    conv2 = sdn.residual_layer('Res1', conv1, 3, 32, 1, phase_train=phase_train, BN=False, relu=False, DSC=True)

    # The third layer
    conv3 = sdn.residual_layer('Res2', conv2, 3, 64, 1, phase_train=phase_train, BN=True, relu=True, DSC=True)

    # Insert inception/residual layer here.
    conv4 = sdn.inception_layer('Inception1', conv3, 64, 2, phase_train=phase_train, BN=False, relu=False)

    # The 4th layer
    conv5 = sdn.residual_layer('Res3', conv4, 3, 256, 1, phase_train=phase_train, BN=True, relu=True, DSC=True)

    # The affine transform layer here:
    h_trans = sdn.spatial_transform_layer('Transformer', conv5)

    # The 5th layer
    conv6 = sdn.convolution('Conv6', h_trans, 3, 512, 1, phase_train=phase_train, downsample=True)

    # The Fc7 layer
    fc7 = sdn.fc7_layer('FC7', conv6, 128, True, phase_train, FLAGS.dropout_factor, BN=False, override=3)

    # Fc8 layer
    fc8 = sdn.linear_layer('fc8', fc7, 32, False, phase_train, BN=False, relu=True)

    # The linear layer: diff is +Relu -slice, +xaviaer +bias zero
    Predictions = sdn.linear_layer('Output', fc8, 1, phase_train=phase_train, relu=False)

    # Retreive the weights collection
    weights = tf.get_collection('weights')

    # Sum the losses
    L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), FLAGS.l2_gamma)

    # Add it to the collection
    tf.add_to_collection('losses', L2_loss)

    # Activation summary
    tf.summary.scalar('L2_Loss', L2_loss)
    print (conv6, fc7)

    return Predictions, L2_loss  # Return whatever the name of the final logits variable is


def total_loss(logits, labels):

    """
    This function adds the loss up
    :param logits: calculated transformations
    :param labels: actual transformations
    :return: MSE_loss the L2 loss
    """

    # Must squeeze because otherwise we are subtracting a row vector from a column vector giving a matrix
    labels = tf.squeeze(labels)
    logits = tf.squeeze(logits)

    # For age sensitive mask based on the average age of 9
    mask = tf.cast(labels, tf.float32)

    # Now normalize so that an age of 0 or 18 gets 2 while age of 9 ges 1
    mask = tf.add(tf.multiply(tf.divide(tf.abs(tf.subtract(12.0, mask)), 5.0), FLAGS.loss_factor), 1.0)

    # Convert to row vector
    mask = tf.squeeze(mask)

    # Calculate MSE with the factor multiplied in
    MSE_loss = tf.reduce_mean(tf.multiply(tf.square(labels-logits), mask))
    #MSE_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, logits)))


    # Output the summary of the MSE and MAE
    tf.summary.scalar('Square Error', MSE_loss)
    tf.summary.scalar('Absolute Error', tf.reduce_mean(tf.abs(labels - logits)))

    # Add these losses to the collection
    tf.add_to_collection('losses', MSE_loss)

    # For now return MSE loss, add L2 regularization below later
    return MSE_loss


def backward_pass(total_loss):
    """
    This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=1e-8)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # clip the gradients
    #gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Does nothing. placeholder to control the execution of the graph
    with tf.control_dependencies([train_op, variable_averages_op]): dummy_op = tf.no_op(name='train')

    return dummy_op


def Inputs(skip=True):
    """
    process input order
    :param skip: whether to skip the preprocessing steps if already have a protobuf
    :return:
    """

    if not skip:

        pre_process_RSNA(gender=FLAGS.gender, dims=FLAGS.dims, xvals=FLAGS.cross_validations)

    data = sdl.randomize_batches(load_protobuf(), FLAGS.batch_size)
    validation = sdl.val_batches(load_validation_set(), FLAGS.batch_size)

    return data, validation


def load_validation_set():

    """
    Loads protocol buffer
    :param return_dict:
    :return:
    """

    # retreive file list
    filenames1 = sdl.retreive_filelist('tfrecords', path='data/')

    # The real filenames
    filenames = []

    # Retreive only the right filename
    for i in range(0, len(filenames1)):
        if str(FLAGS.validation_file) in filenames1[i]:
            filenames.append(filenames1[i])

    print('Test Files: %s' % filenames)

    # now load the remaining files
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)

    reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = reader.read(filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse

    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'data': tf.FixedLenFeature([], tf.string),
                    'reading': tf.FixedLenFeature([], tf.string), 'sex': tf.FixedLenFeature([], tf.string),
                    'age': tf.FixedLenFeature([], tf.string), 'ptid': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Change the raw image data to 8 bit integers first
    image = tf.decode_raw(features['data'], tf.float32)  # Set this examples image to a blank tensor with integer data
    image = tf.reshape(image, shape=[FLAGS.dims, FLAGS.dims, 1])  # Set the dimensions of the image ( must equal input dims here)

    # Cast all our data to 32 bit floating point units. Cannot convert string to number unless you use that function
    id = tf.cast(features['id'], tf.float32)
    reading = tf.string_to_number(features['reading'], tf.float32)
    age = tf.string_to_number(features['age'], tf.float32)
    sex = tf.cast(features['sex'], tf.string)
    ptid = tf.string_to_number(features['ptid'], tf.float32)

    # Gender specific
    male = tf.cond(tf.equal(sex, 'M'), lambda:tf.Variable(1.0, trainable=False), lambda:tf.Variable(0.0, trainable=False))
    # male = tf.Variable(tf.equal(sex, 'M'), dtype=tf.float32)
    # male = tf.cast(tf.equal(sex, 'M'), tf.float32)

    # create float summary image
    tf.summary.image('Testing Image', tf.reshape(image, shape=[1, FLAGS.dims, FLAGS.dims, 1]), max_outputs=4)

    # Now the final resize to network dimensions
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])
    #image = tf.image.per_image_standardization(image)

    # Return data as a dictionary by default
    final_data = {'image': image, 'reading': reading, 'age': age, 'sex': sex, 'male':male, 'ptid':ptid}
    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict


def load_protobuf():
    """
    Same as load protobuf() but loads the validation set
    :return:
    """

    # Load all the filenames in glob
    filenames = sdl.retreive_filelist('tfrecords', path='data/')

    # Define the filenames to remove
    for i in range(0, len(filenames)):
        if str(FLAGS.validation_file) in filenames[i]:
            valid = filenames[i]

    # Delete them from the filename queue
    filenames.remove(valid)
    print('Training Files: %s' % filenames)

    # Load the filename queue
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    val_reader = tf.TFRecordReader()  # Instantializes a TFRecordReader which outputs records from a TFRecords file
    _, serialized_example = val_reader.read(
        filename_queue)  # Returns the next record (key:value) produced by the reader

    # Restore the feature dictionary to store the variables we will retrieve using the parse

    feature_dict = {'id': tf.FixedLenFeature([], tf.int64), 'data': tf.FixedLenFeature([], tf.string),
                    'reading': tf.FixedLenFeature([], tf.string), 'sex': tf.FixedLenFeature([], tf.string),
                    'age': tf.FixedLenFeature([], tf.string), 'ptid': tf.FixedLenFeature([], tf.string)}

    # Parses one protocol buffer file into the features dictionary which maps keys to tensors with the data
    features = tf.parse_single_example(serialized_example, features=feature_dict)

    # Change the raw image data to 8 bit integers first
    image = tf.decode_raw(features['data'], tf.float32)
    image = tf.reshape(image, shape=[FLAGS.dims, FLAGS.dims, 1])

    # Cast all our data to 32 bit floating point units. Cannot convert string to number unless you use that function
    id = tf.cast(features['id'], tf.float32)
    reading = tf.string_to_number(features['reading'], tf.float32)
    age = tf.string_to_number(features['age'], tf.float32)
    sex = tf.cast(features['sex'], tf.string)
    ptid = tf.string_to_number(features['ptid'], tf.float32)

    # Gender specific
    male = tf.cond(tf.equal(sex, 'M'), lambda: tf.Variable(1.0, trainable=False),
                   lambda: tf.Variable(0.0, trainable=False))
    # male = tf.Variable(tf.equal(sex, 'M'), dtype=tf.float32)
    # male = tf.cast(tf.equal(sex, 'M'), tf.float32)

    # Apply image pre processing here:
    image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))

    # For random rotation, generate a random angle and apply the rotation
    image = tf.contrib.image.rotate(image, tf.random_uniform([1], -0.78, 0.78))

    # Resize images
    crop_dims = int(FLAGS.dims * 1.1)
    image = tf.image.resize_images(image, [crop_dims, crop_dims])

    # Random crop the image to a box 80% of the size
    image = tf.random_crop(image, [FLAGS.dims, FLAGS.dims, 1])

    # create float summary image
    tf.summary.image('Training Image', tf.reshape(image, shape=[1, FLAGS.dims, FLAGS.dims, 1]), max_outputs=4)

    # Now the final resize to network dimensions
    image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims])
    #image = tf.image.per_image_standardization(image)

    # Return data as a dictionary by default
    final_data = {'image': image, 'reading': reading, 'age': age, 'sex': sex, 'male': male, 'ptid': ptid}
    returned_dict = {}
    returned_dict['id'] = id
    for key, feature in final_data.items():
        returned_dict[key] = feature
    return returned_dict


def pre_process_RSNA(gender='S', dims=256, xvals = 5, agez = 0, filez = 'ALL'):

    """
    Load the images
    :param gender: the gender to preprocess
    :param dims: the dimensions of the images to save
    :param xvals: the number of cross validations to generate
    :return:
    """

    # Retreive file names
    filenames = sdl.retreive_filelist('png', True, 'data/RSNAData/')
    label_file = sdl.retreive_filelist('csv', False, 'data/RSNAData/')

    # Retreive labels: '10477': {'age': '6.00', 'boneage': '72', 'male': 'TRUE', 'ptid': '13218', 'sex': 'M'}
    labels = sdl.load_CSV_Dict('id', label_file[0])

    # Summarize
    print('Number of images found: ', len(filenames), 'Labels Found: ', len(labels))

    # Global variables
    index, pts = 0, 0
    data = {}
    display = []

    # Loop through all the files and store them all
    for file in filenames:

        # Retreive the ptid
        ptid = os.path.basename(file).split('.')[0]

        # Set sex to none right now
        sex = 'E'

        # load the labels
        for idx, dic in labels.items():

            # Match up the ptid
            if ptid == dic['ptid']:

                # This is it, apply the variables
                age = float(dic['age'])
                reading = float(dic['age'])
                sex = dic['sex']
                id = int(idx)

                # Exit here
                break

        # Skip non gender matches
        if gender != sex:

            # Check for failiure
            if sex == 'E':
                print('Failed to load label for %s' % file)
                continue

            # Versus just regular wrong sex
            continue

        # If the age doesn't fit, skip
        if agez > 10:
            if age <= 8: continue

        elif agez <= 10:
            if age >= 10: continue

        # Load the image
        image = sdl.load_image(file)

        # resize the image
        image = sdl.zoom_2D(image, [dims, dims])

        # Re normalize the image
        image = sdl.normalize(image, True, 0.05)

        # Clip large pixel values, usually the metallic labels which can be up to 14
        image[image>4.5] = 4.5

        # increment patient counter
        pts += 1

        # how many examples of this to create
        examples_per = 1

        # Augment young ages for the imbalanced training set
        if age <= 5: examples_per = min(int(examples_per / (age/10)), 15)

        # Save x examples
        for z in range (examples_per):

            # Create the dictionary
            data[index] = {'age': age, 'reading': reading, 'sex': sex, 'ptid': ptid, 'data': image}

            # Increment index
            index += 1

        # Display summary message
        if pts % 250 == 0: print ('%s Patients loaded, %s examples generated' %(pts, index))

    # Finished all patients
    print ('Patients loaded: %s, Examples Saved: %s, Gender: %s' %(pts, index, filez))

    # Now create a protocol buffer
    print('Creating final protocol buffer... %s entries' %len(data))

    # Initialize normalization images array
    normz = np.zeros(shape=(len(data), dims, dims), dtype=np.float32)

    # Normalize all the images. First retreive the images
    for key, dict in data.items(): normz[key, :, :] = dict['data']

    # Now normalize the whole batch
    print('Batch Norm: %s , Batch STD: %s' % (np.mean(normz), np.std(normz)))
    normz = sdl.normalize(normz, True, 0.1)

    # Return the normalized images to the dictionary
    for key, dict in data.items(): dict['data'] = normz[key]

    # generate x number of writers depending on the cross validations
    writer = []

    # Open the file writers
    for z in range(xvals):

        # Define writer name
        filename = ('data/%sRSNAData%s.tfrecords' %(filez, z))
        writer.append(tf.python_io.TFRecordWriter(filename))

    # Loop through each example and append the protobuf with the specified features
    z = 0
    for key, values in data.items():

        # Serialize to string
        example = tf.train.Example(features=tf.train.Features(feature=sdl.create_feature_dict(values, key)))

        # Save this index as a serialized string in the protobuf
        writer[(z % xvals)].write(example.SerializeToString())
        z += 1

    # Close the file after writing
    for y in range(xvals): writer[y].close()
