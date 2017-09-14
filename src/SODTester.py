"""
SOD Tester contains function wrappers for various metric and performance analysis functions including

mean absolute error, mean squared error, DICE score, sensitivity, specificity, AUC

"""

import os, glob

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import pandas as pd

from scipy import interp


class SODTester():

    """
    SOD Tester class is a class for testing the performance of our network
    """

    # Linear regression variables
    MAE = 0
    best_MAE = 10
    accuracy = 0

    # Classification variables
    TP, FP, TN, FN = 0, 0, 0, 0
    sensitiviy, specificity = 0, 0
    PPV, NPV = 0, 0
    F1_score, AUC = 0, 0
    roc_auc, fpr, tpr = {}, {}, {}

    # Other variables
    right, total, calls = 0, 0, 0
    best_step, num_classes = 0, 0

    def __init__(self, binary, regression):

        # Define whether this is a binary, multiclass or linear regression model
        self.binary = binary
        self.regression = regression

    """
     Performance Metrics
    """

    def calculate_mean_absolute_error(self, prediction, labels, display=True):

        """
        Calculates the MAE between predictions and labels
        :param prediction: network output
        :param labels: the ground truth
        :param display: Whether to print the MAE
        :return: MAE the mae
        """

        # Convert to numpy arrays
        predictions = np.squeeze(prediction.astype(np.float32))
        label = np.squeeze(labels.astype(np.float32))

        # How many to print
        to_print = min(len(label), 15)

        # Calculate MAE
        MAE = np.mean(np.absolute((predictions - label)))

        # Print the summary
        np.set_printoptions(precision=1)  # use numpy to print only the first sig fig
        if display: print('MAE: %s\n Pred: %s\nReal: %s' % (MAE, predictions[:to_print], label[:to_print]))

        # Append right
        self.right += MAE
        self.total += 1

        return MAE


    def retreive_accuracy_regression(self, Epoch, display=True):

        """
        Retreive the accuracy so far
        :param Epoch: what epoch we're in
        :param display: Whether to print
        :return: accuracy
        """

        # Calculate final MAE and ACC
        self.accuracy = self.right / self.total

        # Print the final accuracies and MAE if requested
        if display:
            print('-' * 70)
            print('--- EPOCH: %s MAE: %.3f (Old Best: %.3f @ %s) ---'
                    % (Epoch, self.accuracy, self.best_MAE, self.best_step))

        # Update bests
        if self.accuracy <= self.best_MAE: self.best_step, self.best_MAE = Epoch, self.MAE

        return self.accuracy


    def calculate_metrics(self, logits, labels, positive_class, step, display=True):

        """
        Retreive the Accuracy, Sensitivity, Specificity
        :param logits: network predictions
        :param labels: the ground truth
        :param positive_class: the class we will consider positive for calculating ground truth
        :param step: what step we're on
        :param display: whether to print examples
        :return: acc, sn, sp
        """

        # Retreive and print the labels and logits
        label = np.squeeze(labels.astype(np.int8))
        logit = np.squeeze(np.argmax(logits.astype(np.float), axis=1))

        # First calculate AUC
        self.AUC += skm.roc_auc_score(label, logit)
        self.calls += 1

        # Retreive metrics
        for z in range(len(label)):

            # If we got this right, make it right
            if label[z] == logit[z]: self.right += 1

            # Metrics for when the ground truth is positive
            if label[z] == positive_class:

                # Get metrics
                if label[z] == logit[z]: self.TP += 1
                if label[z] != logit[z]: self.FN += 1

            # Metrics for when the ground truth is negative
            if label[z] != positive_class:

                # Get metrics
                if label[z] == logit[z]: self.TN += 1
                if label[z] != logit[z]: self.FP += 1

        # Increment total
        self.total += len(label)

        # Print Summary if wanted
        if display:

            # How many to print
            to_print = min(len(label), 15)

            # Now print
            print('-' * 70)
            print('Patient %s Class: %s' % (step, label[:to_print]))
            print('Patient %s Preds: %s' % (step, logit[:to_print]))


    def retreive_metrics_classification(self, Epoch, display=True):

        """
        Retreives sn, sp, PPV, NPV, ROC
        :param Epoch: What epoch we're in
        :param display: Whether to print the results
        :return: 
        """

        # Calculate the metrics. To prevent division by zero, use error handling

        # Some metrics only make sense for binary
        if self.binary:
            try: self.sensitiviy = self.TP / (self.TP + self.FN)
            except: self.sensitiviy = 0

            try: self.specificity = self.TN / (self.TN + self.FP)
            except: self.specificity = 0

            try: self.PPV = self.TP / (self.TP + self.FP)
            except: self.PPV = 0

            try: self.NPV = self.TN / (self.TN + self.FN)
            except: self.PPV = 0

            # F1 score
            try: self.F1_score = 2/((1/self.sensitiviy)+(1/self.PPV))
            except: self.F1_score = 0

        # AUC
        self.AUC /= self.calls

        # Accuracy
        self.accuracy = 100 * self.right/self.total

        # Print the final accuracies and MAE if requested
        if display:

            print('-' * 70)

            # Print depends on binary or not
            if self.binary:

                print('--- EPOCH: %s, ACC: %.2f, SN: %.3f, SP: %.3f, AUC: %.3f, F1: %.3f ---'
                      % (Epoch, self.accuracy, self.sensitiviy, self.specificity, self.AUC, self.F1_score))
                print('--- True Pos: %s, False Pos: %s, True Neg: %s, False Neg: %s ---'
                      % (self.TP, self.FP, self.TN, self.FN))

            else:

                # Display one per class
                print('--- EPOCH: %s, ACC: %.2f %%, AUC: %.3f, ---' % (Epoch, self.accuracy, self.AUC))
                for z in range(self.num_classes): print('Class %s: %.3f --- ' % (z, self.roc_auc[z]), end='')
                print('Micro AUC: %.3f, Macro AUC: %.3f' % (self.roc_auc['micro'], self.roc_auc["macro"]))


    def save_to_csv(self, patients, predictions, step, error, filename='submission'):
        """
        Saves the patient dictionaries to a CSV
        :param patients: the dictionary of patient indexes
        :param predictions: the predicted outputs of the network
        :param step: the current step. If not 0 then we will be appending the csv not making a new one
        :param error: the column with the error
        :param filename: the filename to save
        :return:
        """

        # Make dummy dictionary
        dictionary = {}

        # Loop through the dictionary
        for idx, array in patients.items():

            # no need to save the image data
            if idx == 'image': continue

            # append new dictionary
            dictionary[idx] = np.squeeze(np.array(array))

        # Append the predictions
        dictionary['predictions'] = predictions
        dictionary['error'] = error

        # Now create the data frame and save the csv
        df = pd.DataFrame(dictionary)

        # Append if this is not the first step
        if step != 0:
            with open(filename, 'a') as f: df.to_csv(f, index=True, index_label='Batch_Num', header=False)

        # Otherwise make a new CSV
        else: df.to_csv(filename, index=True, index_label='Batch_Num')


    def calculate_boneage_errors(self, predictions, label):
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
            if FLAGS.model < 3:  # Girls
                if label[i] <= (3 / 12):
                    std_dev[i] = 0.72 / 12
                elif label[i] <= (6 / 12):
                    std_dev[i] = 1.16 / 12
                elif label[i] <= (9 / 12):
                    std_dev[i] = 1.36 / 12
                elif label[i] <= (12 / 12):
                    std_dev[i] = 1.77 / 12
                elif label[i] <= (18 / 12):
                    std_dev[i] = 3.49 / 12
                elif label[i] <= (24 / 12):
                    std_dev[i] = 4.64 / 12
                elif label[i] <= (30 / 12):
                    std_dev[i] = 5.37 / 12
                elif label[i] <= 3:
                    std_dev[i] = 5.97 / 12
                elif label[i] <= 3.5:
                    std_dev[i] = 7.48 / 12
                elif label[i] <= 4:
                    std_dev[i] = 8.98 / 12
                elif label[i] <= 4.5:
                    std_dev[i] = 10.73 / 12
                elif label[i] <= 5:
                    std_dev[i] = 11.65 / 12
                elif label[i] <= 6:
                    std_dev[i] = 10.23 / 12
                elif label[i] <= 7:
                    std_dev[i] = 9.64 / 12
                elif label[i] <= 8:
                    std_dev[i] = 10.23 / 12
                elif label[i] <= 9:
                    std_dev[i] = 10.74 / 12
                elif label[i] <= 10:
                    std_dev[i] = 11.73 / 12
                elif label[i] <= 11:
                    std_dev[i] = 11.94 / 12
                elif label[i] <= 12:
                    std_dev[i] = 10.24 / 12
                elif label[i] <= 13:
                    std_dev[i] = 10.67 / 12
                elif label[i] <= 14:
                    std_dev[i] = 11.3 / 12
                elif label[i] <= 15:
                    std_dev[i] = 9.23 / 12
                else:
                    std_dev[i] = 7.31 / 12

            else:  # Boys
                if label[i] <= (3 / 12):
                    std_dev[i] = 0.72 / 12
                elif label[i] <= (6 / 12):
                    std_dev[i] = 1.13 / 12
                elif label[i] <= (9 / 12):
                    std_dev[i] = 1.43 / 12
                elif label[i] <= (12 / 12):
                    std_dev[i] = 1.97 / 12
                elif label[i] <= (18 / 12):
                    std_dev[i] = 3.52 / 12
                elif label[i] <= (24 / 12):
                    std_dev[i] = 3.92 / 12
                elif label[i] <= (30 / 12):
                    std_dev[i] = 4.52 / 12
                elif label[i] <= 3:
                    std_dev[i] = 5.08 / 12
                elif label[i] <= 3.5:
                    std_dev[i] = 5.40 / 12
                elif label[i] <= 4:
                    std_dev[i] = 6.66 / 12
                elif label[i] <= 4.5:
                    std_dev[i] = 8.36 / 12
                elif label[i] <= 5:
                    std_dev[i] = 8.79 / 12
                elif label[i] <= 6:
                    std_dev[i] = 9.17 / 12
                elif label[i] <= 7:
                    std_dev[i] = 8.91 / 12
                elif label[i] <= 8:
                    std_dev[i] = 9.10 / 12
                elif label[i] <= 9:
                    std_dev[i] = 9.0 / 12
                elif label[i] <= 10:
                    std_dev[i] = 9.79 / 12
                elif label[i] <= 11:
                    std_dev[i] = 10.09 / 12
                elif label[i] <= 12:
                    std_dev[i] = 10.38 / 12
                elif label[i] <= 13:
                    std_dev[i] = 10.44 / 12
                elif label[i] <= 14:
                    std_dev[i] = 10.72 / 12
                elif label[i] <= 15:
                    std_dev[i] = 11.32 / 12
                elif label[i] <= 16:
                    std_dev[i] = 12.86 / 12
                else:
                    std_dev[i] = 13.05 / 12

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
