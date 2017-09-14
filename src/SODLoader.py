"""
SOD Loader is the class for loading various file types including: JPegs Nifty and DICOM into numpy arrays.

There are also functions to preprocess the data including: segmenting lungs, generating cubes, and creating MIPs

It then contains functions to store the file as a protocol buffer

"""

import glob, os, dicom, csv, random, cv2, math, astra

import numpy as np
import nibabel as nib
import tensorflow as tf
import SimpleITK as sitk
import scipy.ndimage as scipy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from scipy.io import loadmat
from skimage import morphology


class SODLoader():

    """
    SOD Loader class is a class for loading all types of data into protocol buffers
    """

    def __init__(self, data_root):

        """
        Initializes the class handler object
        :param data_root: The source directory with the data files
        """

        self.data_root = data_root
        self.files_in_root = glob.glob('**/*', recursive=True)

        # Data is all the data, everything else is instance based
        self.data = {}

        # Stuff for the dictionary
        self.label = None
        self.image = None
        self.origin = None
        self.spacing = None
        self.dims = None
        self.patient = None     # Usually the accession number


    """
     Data Loading Functions. Support for DICOM, Nifty, CSV
    """


    def load_DICOM_2D(self, path, dtype=np.int16):

        """
        This function loads a 2D DICOM file and stores it into a numpy array. From Bone Age
        :param path: The path of the DICOM files
        :param: dtype = what data type to save the image as
        :return: image = A numpy array of the image
        :return: accno = the accession number
        :return: dims = the dimensions of the image
        :return: window = the window level of the file
        """

        # Load the Dicom
        try:
            ndimage = dicom.read_file(path)
        except:
            print('For some reason, cant load: %s' % path)
            return

        # Retreive the dimensions of the scan
        dims = np.array([int(ndimage.Columns), int(ndimage.Rows)])

        try:
            # Retreive the window level
            window = [int(ndimage.WindowCenter), int(ndimage.WindowWidth)]
        except: window = None

        # Retreive the dummy accession number
        accno = int(ndimage.AccessionNumber)

        # Finally, make the image actually equal to the pixel data and not the header
        image = np.asarray(ndimage.pixel_array, dtype)

        # Convert to Houndsfield units if slope and intercept is available:
        try:
            # retreive the slope and intercept of this slice
            slope = ndimage.RescaleSlope
            intercept = ndimage.RescaleIntercept

            # If the slope isn't 1, rescale the images using the slope
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(dtype)

            # Reset the Intercept
            image += dtype(intercept)
        except: pass

        return image, accno, dims, window


    def load_CSV_Dict(self, indexname, path):
        """
        This function loads the annotations into a dictionary of dictionary with the columns as keys
        :param indexname: what column name to assign as the index for each dictionary
        :param path: file name
        :return: return_dict: a dict of dicts with indexname as the pointer and each entry based on the title row
        """

        # Create the reader object to load
        reader = csv.DictReader(open(path))

        # Initialize the return dictionary
        return_dict = {}

        # Iterate and append the dictionary
        for row in reader:

            # Name the key as the indexname
            key = row.pop(indexname)

            # What to do if there is a duplicate: rename with 1 at the end,
            if key in return_dict:
                key = key + '1'

                # For another duplicate, do the same
                if key in return_dict:
                    key = key + '2'

            # Make the entire row (as a dict) the index
            return_dict[key] = row

        return return_dict


    def load_NIFTY(self, path, reshape=True):
        """
        This function loads a .nii.gz file into a numpy array with dimensions Z, Y, X, C
        :param filename: path to the file
        :param reshape: whether to reshape the axis from/to ZYX
        :return:
        """

        #try:

        # Load the data from the nifty file
        raw_data = nib.load(path)

        # Reshape the image data from NiB's XYZ to numpy's ZYXC
        if reshape: data = self.reshape_NHWC(raw_data.get_data(), False)
        else: data = raw_data.get_data()

        # Return the data
        return data


    def load_image(self, path, grayscale=True):
        """
        Loads an image from a jpeg or other type of image file into a numpy array
        :param path:
        :param grayscale:
        :return:
        """

        # Make the image into a greyscale by default
        if grayscale:
            image = mpimg.imread(path)

        # Else keep all the color channels
        else:
            image = mpimg.imread(path)

        return image


    def randomize_batches(self, image_dict, batch_size):
        """
        This function takes our full data tensors and creates shuffled batches of data.
        :param image_dict: the dictionary of tensors with the images and labels
        :param batch_size: batch size to shuffle
        :return: 
        """

        min_dq = 16  # Min elements to queue after a dequeue to ensure good mixing
        capacity = min_dq + 3 * batch_size  # max number of elements in the queue
        keys, tensors = zip(*image_dict.items())  # Create zip object

        # This function creates batches by randomly shuffling the input tensors. returns a dict of shuffled tensors
        shuffled = tf.train.shuffle_batch(tensors, batch_size=batch_size,
                                          capacity=capacity, min_after_dequeue=min_dq)

        # Dictionary to store our shuffled examples
        batch_dict = {}

        # Recreate the batched data as a dictionary with the new batch size
        for key, shuffle in zip(keys, shuffled): batch_dict[key] = shuffle

        return batch_dict


    def val_batches(self, image_dict, batch_size):

        """
        Loads a validation set without shuffling it
        :param image_dict: the dictionary of tensors with the images and labels
        :param batch_size: batch size to shuffle
        :return:
        """

        min_dq = 16  # Min elements to queue after a dequeue to ensure good mixing
        capacity = min_dq + 3 * batch_size  # max number of elements in the queue
        keys, tensors = zip(*image_dict.items())  # Create zip object

        # This function creates batches by randomly shuffling the input tensors. returns a dict of shuffled tensors
        shuffled = tf.train.batch(tensors, batch_size=batch_size, capacity=capacity)

        batch_dict = {}  # Dictionary to store our shuffled examples

        # Recreate the batched data as a dictionary with the new batch size
        for key, shuffle in zip(keys, shuffled): batch_dict[key] = shuffle

        return batch_dict


    """
             Pre processing functions.
    """

    def zoom_2D(self, image, new_shape):
        """
        Uses open CV to resize a 2D image
        :param image: The input image, numpy array
        :param new_shape: New shape, tuple or array
        :return: the resized image
        """
        return cv2.resize(image,(new_shape[0], new_shape[1]), interpolation = cv2.INTER_CUBIC)


    """
         Utility functions: Random tools for help
    """

    def normalize(self, input, crop=False, crop_val=0.5):
        """
        Normalizes the given np array
        :param input:
        :param crop: whether to crop the values
        :param crop_val: the percentage to crop the image
        :return:
        """

        if crop:

            ## CLIP top and bottom x values and scale rest of slice accordingly
            b, t = np.percentile(input, (crop_val, 100-crop_val))
            slice = np.clip(input, b, t)
            if np.std(slice) == 0:
                return slice
            else:
                return (slice - np.mean(slice)) / np.std(slice)

        return (input - np.mean(input)) / np.std(input)


    def reshape_NHWC(self, vol, NHWC):
        """
        Method to reshape 2D or 3D tensor into Tensorflow's NHWC format
        vol: The image data
        NHWC whether the input has a channel dimension
        """

        # If this is a 2D image
        if len(vol.shape) == 2:

            # Create an extra channel at the beginning
            vol = np.expand_dims(vol, axis=0)

        # If there are 3 dimensions to the shape (2D with channels or 3D)
        if len(vol.shape) == 3:

            # If there is no channel dimension (i.e. grayscale)
            if not NHWC:

                # Move the last axis (Z) to the first axis
                vol = np.moveaxis(vol, -1, 0)

            # Create another axis at the end for channel
            vol = np.expand_dims(vol, axis=3)


        return vol


    def gray2rgb(self, img, maximum_val=1, percentile=0):
        """
        Method to convert H x W grayscale tensor to H x W x 3 RGB grayscale
        :params
        (np.array) img : input H x W tensor
        (int) maximum_val : maximum value in output
          if maximum_val == 1, output is assumed to be float32
          if maximum_val == 255, output is assumed to be uint8 (standard 256 x 256 x 256 RGB image)
        (int) percentile : lower bound to set to 0
        """
        img_min, img_max = np.percentile(img, percentile), np.percentile(img, 100 - percentile)
        img = (img - img_min) / (img_max - img_min)
        img[img > 1] = 1
        img[img < 0] = 0
        img = img * maximum_val
        img = np.expand_dims(img, 2)
        img = np.tile(img, [1, 1, 3])

        dtype = 'float32' if maximum_val == 1 else 'uint8'
        return img.astype(dtype)


    def retreive_filelist(self, extension, include_subfolders=False, path=None):
        """
        Returns a list with all the files of a certain type in path
        :param extension: what extension to search for
        :param include_subfolders: whether to include subfolders
        :param path: specified path. otherwise use data root
        :return:
        """

        # If they want to return the folder list, do that
        if extension == '*': return glob.glob(path + '*')

        # If no path specified use the default data root
        if not path: path = self.data_root

        # If we're including subfolders
        if include_subfolders: extension = ('**/*.%s' % extension)

        # Otherwise just search this folder
        else: extension = ('*.%s' %extension)

        # Join the pathnames
        path = os.path.join(path, extension)

        # Return the list of filenames
        return glob.glob(path, recursive=include_subfolders)


    """
         Tool functions: Most of these are hidden
    """

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def create_feature_dict(self, data_to_write={}, id=1):
        """
        Create the features of each image:label pair we want to save to our TFRecord protobuf here instead of inline
        :param data_to_write: The data we will be writing into a dict
        :param id: The ID of this entry
        :return:
        """

        # initialize an empty dictionary
        feature_dict_write = {}

        # id is the unique identifier of the image, make it an integer
        feature_dict_write['id'] = self._int64_feature(int(id))

        # Loop over the dictionary and append the feature list for each id
        for key, feature in data_to_write.items():

            # If this is our Data array, use the tostring() method.
            if key == 'data':
                feature_dict_write[key] = self._bytes_feature(feature.tobytes())  #

            else:  # Otherwise convert to a string and encode as bytes to pass on
                features = str(feature)
                feature_dict_write[key] = self._bytes_feature(features.encode())

        return feature_dict_write
