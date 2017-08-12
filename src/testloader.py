import csv
import cv2
import dicom
import glob
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_annotations(filename, indexname='Patient'):
    """
    This function loads the annotations into a dictionary of dictionary with the columns as keys
    Args:
        filename: The file name
        indexname: what column name to assign as the index for each dictionary
    Returns:
        A dictionary of dictionaries with the format: index: { columnname: value}
    """

    reader = csv.DictReader(open(filename))  # Create the reader object to load
    return_dict = {}  # Initialize the return dictionary

    # Iterate and append the dictionary
    for row in reader:
        key = row.pop(indexname)  # Name the key as the indexname
        return_dict[key] = row  # Make the entire row (as a dict) the index

    return return_dict


def load_labels(filename, indexname='ACC'):
    """
    This function loads the annotations into a dictionary of dictionary with the columns as keys
    Args:
        filename: The file name
        indexname: what column name to assign as the index for each dictionary
    Returns:
        A dictionary of dictionaries with the format: index: { columnname: value}
    """

    reader = csv.DictReader(open(filename))  # Create the reader object to load
    return_dict = {}  # Initialize the return dictionary

    # Iterate and append the dictionary
    for row in reader:
        key = row.pop(indexname)  # Name the key as the indexname
        return_dict[key] = row  # Make the entire row (as a dict) the index

    return return_dict


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


def load_DICOM(path):
    """
    This function loads a DICOM file and stores it into a numpy array
    :param path: The path of the DICOM files
    :return: ndimage = A numpy array of the image
    :return: accno = the "dummy" accession number
    """

    # Load the Dicom
    try:
        ndimage = dicom.read_file(path)
    except:
        print ('For some reason, cant load: %s' %path)
        return

    # Retreive the dimensions of the scan
    dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])

    # Retreive the window level
    # window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]

    # Retreive the dummy accession number
    accno = int(ndimage.AccessionNumber)

    # Finally, make the image actually equal to the pixel data and not the header
    image = np.asarray(ndimage.pixel_array, np.int16)

    # Convert to Houndsfield units if slope and intercept is available:
    try:
        # retreive the slope and intercept of this slice
        slope = ndimage.RescaleSlope
        intercept = ndimage.RescaleIntercept

        # If the slope isn't 1, rescale the images using the slope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        # Reset the Intercept
        image += np.int16(intercept)
    except:
        pass

    return image, accno, dims


def getNumWord(math_str1):
    """
    This function returns the FIRST given year and month reading of the string passed
    :param math_str1: The string passed, must be all uppercase
    :return: month, year
    """

    # First define the allowed numbers
    allowed = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN']

    # Replace dashes or slashes with spaes
    math_str1 = math_str1.replace('-', ' ')

    # Split the string into an array of strings of one word each
    math_str = math_str1.split()

    # define integers that are clearly wrong (for error checking later)
    yrs = 100
    months = 100

    # For enumeration
    stre = math_str

    # Changes all the lettered numbers to integer strings
    for i, val in enumerate(math_str):
        if val in allowed:
            for j, val2 in enumerate(allowed):
                if val == val2: stre[i] = str(j)

    # Now loop through and make month = the number before the word month
    for i, val in enumerate(stre):
        if 'MONTH' in val:
            months = stre[i-1]
            break # break in case there are more

        # Skip all the months named after the std dev
        if 'DEVIATION' in val: break

    # and same for year
    for i, val in enumerate(stre):
        if 'YEAR' in val:
            yrs = stre[i-1]
            break # break in case there are more

    if isinstance(yrs, int): # Something effed up
        try: yrs = re.search(r'\d+', math_str1).group()
        except: yrs = 'missing'

    if isinstance(months, int): months = 0 # If month was unchanged then it means it's zero

    #print ('Edited: %s, yr: %s, month: %s' % (stre, yrs, months))

    # Make final reading a decimal
    mn = float(months)/12
    try:
        final = float(yrs) + mn
    except: # Must be a range case, retrieve the first int
        try: yrs = re.search(r'\d+', yrs).group() # Retreive first int
        except: print ('range case %s' % math_str1)
        final = float(yrs) + mn # Add the months
        final = (final + (final + 1)) / 2   # Get average after adding a year

    return final


def edit_annotations(annotations1, path):

    # Create copy of annotations with retained values
    annotations = []
    i = 0  # counter

    # Retreive the reading
    for key, dict in annotations1.items():

        # Skip labels without a file
        if dict['ID'] == 'None': continue

        # Make the string upper case
        stry = dict['FindingsOnly'].upper()

        # Holder
        other = ('zzz - ' + dict['FindingsOnly'])

        # Nested try statements to handle errors
        try:
            # Try splitting by the word bone
            stry = stry.split('BONE', 1)[1]

            # Retreive the years
            years = getNumWord(stry)

        except:
            try:
                # Failed? try using the word skeletal
                stry = stry.split('SKELETAL', 1)[1]
                years = getNumWord(stry)

            except:
                # Failed? We will just have to manually fix these
                # print (key, dict['FindingsOnly'])

                # Set years to missing to alert us
                years = 'Missing'
                other = dict['FindingsOnly']

        # Calculate the age here
        dob = datetime.strptime(dict['DOB'], "%m/%d/%Y")
        doe = datetime.strptime(dict['DONE'], "%m/%d/%Y")

        # Calculate age as the difference in days divided by 365. Truncate it
        age = float('%.2f' % (abs((doe - dob).days) / 365))

        # Find abnormal values
        try:
            norm_diff = abs(age - years)
            if norm_diff > age / 5: other = dict['FindingsOnly']
        except:
            norm_diff = 'Invalid'

        # Add the dictionary entry for this patient
        annotations.append({'ACC': key, 'ID': dict['ID'], 'SEX': dict['SEX'],
                            'AGE': age, 'Reading': years, 'Diff': norm_diff, 'Findings': other})

        # Testing Code
        i += 1
        # if i> 1000: break

    # Initialize the CSV file handler
    csvf = open((path+'labeldata.csv'), 'w')

    # Create the CSV column names
    csvcolumns = ['ACC', 'ID', 'SEX', 'AGE', 'Reading', 'Diff', 'Findings']

    # Now open the CSV file
    csvfile = csv.DictWriter(csvf, fieldnames=csvcolumns)
    csvfile.writeheader()

    # Write the data to the CSV file
    for dicdic in annotations:
        csvfile.writerow(dicdic)

    # Close the CSV handler
    csvf.close()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_feature_dict(data_to_write={}, id=1, restore=False):
    """ Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline"""

    if not restore:  # Do this for the storage dictionary first
        feature_dict_write = {}  # initialize an empty dictionary
        feature_dict_write['id'] = _int64_feature(int(id))  # id is the unique identifier of the image
        for key, feature in data_to_write.items():  #  append the feature list for each id

            # If this is our Data array, use the tostring() method.
            if key == 'data':
                feature_dict_write[key] = _bytes_feature(feature.tobytes())  #

            else:  # Otherwise convert to a string and encode as bytes to pass on
                features = str(feature)
                feature_dict_write[key] = _bytes_feature(features.encode())

        return feature_dict_write

    else:  # Else do this to create the restoration dictionary
        feature_dict_restore = {'id': tf.FixedLenFeature([], tf.int64)}
        for key, feature in data_to_write.items():
            feature_dict_restore[key] = tf.FixedLenFeature([], tf.string)

        return feature_dict_restore


def write_protobuf(examples, cross_validations, name):
    """
    Writes the cumc examples into a protocol buffer
    :param examples:
    :param cross_val: The number of cross validation batches
    :return:
    """

    # Define the array we will use to write the records to the .tfrecords protobuf.
    writer = []

    # Initialize all the writers
    for i in range(0, cross_validations):
        # Define the filenames
        filenames = os.path.join(name + 'cumcboneage' + str(i) + '.tfrecords')

        # Open the file writer
        writer.append(tf.python_io.TFRecordWriter(filenames))

    # Variable to keep track of how many files we skipped
    skipped = 0

    # Variable to keep track of how many files were loaded
    loaded = 0

    # Loop through each example and append the protobuf with the specified features
    for index, feature in examples.items():

        # Create our dictionary of values to store: Added some dimensions values that may be useful later on
        data = {'age': feature['age'], 'reading': feature['reading'], 'sex': feature['sex'],
                'id': index, 'data': feature['data']}

        example = tf.train.Example(features=tf.train.Features(feature=create_feature_dict(data, index)))

        # Calculate the file index as 0 - #cv's
        index = loaded % cross_validations

        # This image made it in increment the loaded image counter
        loaded += 1

        # Save this index as a serialized string in the protobuf
        writer[index].write(example.SerializeToString())

    for i in range(0, cross_validations): writer[i].close()  # Close the file after writing

    # Validation size is total divided by cross validations
    val_size = loaded / cross_validations

    # Print loading info
    print('Skipped: %s non-gender and age matched images, Validation size: %s' % (skipped, val_size))

    return


def get_data(target_age=5, target_gender='M', age_chk=False):


    # Global variables
    path = '/home/stmutasa/PycharmProjects/BoneAge/src/data/cumc/'
    labels = os.path.join(path, 'labeldata.csv')

    # Load the label CSV
    annotations = load_labels(labels, indexname='ACC')

    # Test filename:
    files1 = glob.glob(path + '*')

    # Remove duplicates
    files = list(set(files1))

    # Initialize counters
    failed = 0
    loaded = 0

    # Global variables

    skipped = 0
    total = 0

    # Initialize data dictionary
    data = {}

    for itemz in files:

        # First skip the csv files
        if 'csv' in itemz: continue

        # Then ignore files that aren't in the training directory or files that aren't folders
        if not os.path.isfile(itemz):
            print("Not a file: %s" % itemz)
            failed += 1
            continue

        # Then try load the DICOM
        try:
            image, fake_acc, resolution = load_DICOM(itemz)
        except:
            print ('Failed to load DICOM: %s' %itemz)
            failed += 1
            continue

        # Increment the loaded counter
        loaded += 1

        # Set dummy variables
        id = 0                          # The fake accession number
        reading = 0                     # The bone age reading
        gender = 'F'                    # We all start that way anyway
        age = 0                         # Chronological age

        # Now update the ID if it exists in the label csv
        for idx, dic in annotations.items():

            # Check that the ID matches the fake accno in the DICOM
            if int(dic['ID']) == fake_acc:
                id = int(dic['ID'])
                reading = dic['Reading']
                gender = dic['SEX']
                age = float(dic['AGE'])
                break

        # If ID is still 0 then this file isn't labeled
        if id == 0:
            print('No label exists for %s' %itemz)
            failed += 1
            continue

        # If reading is missing then its missing
        if reading == 'Missing':
            failed += 1
            continue

        # If reading is above 20 then its a mistake
        if float(reading) > 20:
            print('Read erroneous: %s' % itemz)
            failed += 1
            continue

        # Skip the gender not specified by the user
        if target_gender.upper() != gender.upper():
            skipped += 1
            continue

        # If the age doesn't fit, skip
        elif target_age > 10 and age_chk:
            if age <= 9:
                skipped += 1
                continue

        elif target_age <= 10 and age_chk:
            if age >= 10:
                skipped += 1
                continue

        # This one made it through
        total += 1

        # Success, resize the image with openCV
        image = cv2.resize(image.astype('float32'), dsize=(512, 512), interpolation=cv2.INTER_AREA)

        # Dicom max values are 4095. Center image by subtracting half
        image -= 2048

        # Standardize the image
        # image = (image - np.mean(image)) / np.std(image)

        # now set up a dictionary entry
        data[id] = {'age': age, 'reading': float(reading), 'sex': gender, 'data': image}

        # Testing code:
        # if total > 263 : break
        #  Print progress
        if loaded % 1000 == 0:
            print ('%s files loaded so far with %s saved' %(loaded, total))

    print ('Failed: %s, Loaded: %s, Skipped: %s, Total in this Protobuf Set: %s' %(failed, loaded, skipped, total))
    return data


def get_test_data(target_age=5, target_gender='M', age_chk=False):


    # Global variables
    path = '/home/stmutasa/PycharmProjects/BoneAge/src/data/testing/'
    labels = os.path.join(path, 'testlabs.csv')

    # Load the label CSV: '4715798': {'R1': '13', 'AGE': '14.06', 'R2': '10', 'SEX': 'F'}
    annotations = load_labels(labels, indexname='ID')

    # Test filename: '.../BoneAge/src/data/testing/13-14/4630665/_rsa9006__0622012519/ser001img00001.dcm'
    files = glob.glob(path + '**/*.dcm', recursive=True)
    print (len(files), len(annotations))
    # Initialize counters
    failed = 0
    loaded = 0

    # Global variables

    skipped = 0
    total = 0

    # Initialize data dictionary
    data = {}

    for itemz in files:

        # First skip the csv files
        if 'csv' in itemz: continue

        # Then ignore files that aren't in the training directory or files that aren't folders
        if not os.path.isfile(itemz):
            print("Not a file: %s" % itemz)
            failed += 1
            continue

        # Then try load the DICOM
        try:
            image, fake_acc, resolution = load_DICOM(itemz)
        except:
            print ('Failed to load DICOM: %s' %itemz)
            failed += 1
            continue

        # Increment the loaded counter
        loaded += 1

        # Set dummy variables
        id = 0                          # The fake accession number
        reading = 0                     # The bone age reading
        gender = 'F'                    # We all start that way anyway
        age = 0                         # Chronological age

        # Now update the ID if it exists in the label csv
        for idx, dic in annotations.items():

            # Check that the ID matches the fake accno in the DICOM
            if dic['ACC'] in itemz:
                id = idx
                R1 = float(dic['R1'])
                R2 = float(dic['R2'])
                reading = (R1 + R2)/2
                gender = dic['SEX']
                age = float(dic['AGE'])
                break

        # If ID is still 0 then this file isn't labeled
        if id == 0:
            #print('No label exists for %s' %itemz)
            failed += 1
            continue

        # If reading is missing then its missing
        if reading == 'Missing':
            failed += 1
            continue

        # If reading is above 20 then its a mistake
        if float(reading) > 20:
            print('Read erroneous: %s' % itemz)
            failed += 1
            continue

        # Skip the gender not specified by the user
        if target_gender.upper() != gender.upper():
            skipped += 1
            continue

        # If the age doesn't fit, skip
        elif target_age > 10 and age_chk:
            if age <= 9:
                skipped += 1
                continue

        elif target_age <= 10 and age_chk:
            if age >= 10:
                skipped += 1
                continue

        # This one made it through
        total += 1

        # Success, resize the image with openCV
        image = cv2.resize(image.astype('float32'), dsize=(512, 512), interpolation=cv2.INTER_AREA)

        # Dicom max values are 4095. Center image by subtracting half
        image -= 2048

        # Standardize the image
        # image = (image - np.mean(image)) / np.std(image)

        # now set up a dictionary entry
        data[id] = {'age': age, 'reading': float(reading), 'sex': gender, 'data': image}

        # Testing code:
        if loaded % 25 == 0:
            print ('%s files loaded so far with %s saved' %(loaded, total))

    print ('Failed: %s, Loaded: %s, Skipped: %s, Total in this Protobuf Set: %s' %(failed, loaded, skipped, total))
    return data

# data = get_data(5, 'M', True)
# write_protobuf(data, 8, 'YMns')
# data.clear()
#
# data = get_data(15, 'M', True)
# write_protobuf(data, 8, 'OMns')
# data.clear()
# #
# data = get_data(5, 'F', True)
# write_protobuf(data, 8, 'YFns')
# data.clear()
#
# data = get_data(15, 'F', True)
# write_protobuf(data, 8, 'OFns')
# data.clear()

# data = get_data(5, 'M', False)
# write_protobuf(data, 8, 'MAllns')
# data.clear()
#
# data = get_data(5, 'F', False)
# write_protobuf(data, 8, 'FAllns')
# data.clear()

data = get_test_data(5, 'M', True)
write_protobuf(data, 1, 'YMtest')
data.clear()

# data = get_test_data(15, 'M', True)
# write_protobuf(data, 1, 'OMtest')
# data.clear()
#
# data = get_test_data(5, 'F', True)
# write_protobuf(data, 1, 'YFtest')
# data.clear()
#
# data = get_test_data(15, 'F', True)
# write_protobuf(data, 1, 'OFtest')
# data.clear()
#
# data = get_test_data(5, 'M', False)
# write_protobuf(data, 1, 'MAlltest')
# data.clear()
#
# data = get_test_data(5, 'F', False)
# write_protobuf(data, 1, 'FAlltest')
# data.clear()