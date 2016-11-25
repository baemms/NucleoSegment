"""
Classifier to identify nuclei parameters
"""

# general
import numpy as np
import pickle
import os

# classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# validation
from sklearn.cross_validation import cross_val_score

import storage.config as cfg
from storage.image import ImageHandler
from processing.segmentation import Segmentation
from processing.correction import Correction


class Classifier:

    # classifier identifiers
    CLF_SVM = 'SVM'
    CLF_RFC = 'RFC'
    CLF_DTC = 'DTC'

    def __init__(self, segmentation):
        """
        Init classifier parameters

        :param segmentation:
        """
        # set image infos
        self.segmentation = segmentation

        # init classifier
        self.clf = None

        if self.load_classifier() is False:
            self.init_classifier(cfg.clf_method, cfg.clf_estimators)

    def init_classifier(self, clf_type, estimators):
        """
        Init given classifier

        :param clf_type:
        :return:
        """
        # create classifier based on selection
        if clf_type == Classifier.CLF_SVM:
            self.clf = svm.SVC(gamma=0.001, C=estimators, probability=True)
        elif clf_type == Classifier.CLF_RFC:
            self.clf = RandomForestClassifier(n_estimators=estimators)
        elif clf_type == Classifier.CLF_DTC:
            self.clf = DecisionTreeClassifier()

        print('=== %s Classifier ===' % self.clf.__class__.__name__)

    def load_classifier(self):
        """
        Load classifier from experiment

        :return:
        """
        loaded = False

        # load classifier from file
        if os.path.isfile(self.segmentation.get_results_dir().classifier) is True:
            with open(self.segmentation.get_results_dir().classifier, "rb") as fin:
                self.clf = pickle.load(fin)
                loaded = True

        return loaded

    def save_classifier(self):
        """
        Save classifier in experiment

        :return:
        """
        # save classifier to file
        if hasattr(self, 'clf') and self.clf is not None:
            # save object
            with open(self.segmentation.get_results_dir().classifier, "wb") as fin:
                pickle.dump(self.clf, fin)

    def train_with_exts(self):
        """
        Use extensions of experiment to train the classifier

        :return:
        """
        # get extensions from experiment
        self.ext_infos = ImageHandler.get_ext_infos_by_expnum(self.segmentation.image_info['ID'])

        # create train and target vectors
        self.train_data, self.target_data = self.create_train_target_from_ext_infos()

        # create unkown vector for image
        self.unknown_data, self.unknown_map = self.create_unknown()

        # train classifier and get probabilities
        self.nuclei_probas = self.train_and_predict()

        # add probability to nuclei
        self.add_nuclei_probas_to_params()

    def add_nuclei_probas_to_params(self):
        """
        Add nuclei probabilities to params of nuclei

        :return:
        """
        # go through nuclei and add probas
        for i, nucleus_proba in enumerate(self.nuclei_probas):
            self.segmentation.add_nucleus_param(self.unknown_map[i], 'nuc_proba', nucleus_proba[1])

        # save segmentation
        self.segmentation.save()

    def create_train_target_from_ext_infos(self):
        """
        Create training and target vectors from the
        given extensions
        :return:
        """
        # store training and correction
        train_nuclei = list()
        corr_nuclei = list()

        # go through infos
        for info in self.ext_infos:
            # load segmentation
            seg = Segmentation(info)
            seg.load()

            # add to training nuclei
            train_nuclei += seg.nuclei.copy()

            # load correction
            corr = Correction(seg)
            corr.load_corrections()

            # add to correction
            corr_nuclei += corr.corr_nonuc.copy()

        # load train params
        train_params = cfg.clf_train_params

        # create data and target vectors
        train_data = np.zeros(shape=(len(train_nuclei), len(train_params)))
        train_target = np.zeros(shape=(len(train_nuclei)))

        # go through nuclei and build target vector
        for i, nucleus in enumerate(train_nuclei):
            # get relevant params from nucleus
            for j, key in enumerate(train_params):
                train_data[i][j] = nucleus[key]

            # set target
            nucleus_value = 0

            # is it a nucleus?
            if Correction.is_correction_nonuc_with_list(nucleus, corr_nuclei) is False:
                nucleus_value = 1  # yes - it is

            train_target[i] = nucleus_value

        return train_data, train_target

    def create_unknown(self):
        """
        Create unknown vector

        :return:
        """
        # store unknown
        unknown_nuclei = list()
        corr_nuclei = list()

        # add to training nuclei
        unknown_nuclei += self.segmentation.nuclei.copy()

        # load correction
        corr = Correction(self.segmentation)
        corr.load_corrections()

        # add to correction
        if hasattr(corr, 'corr_nonuc') is True:
            corr_nuclei += corr.corr_nonuc.copy()

        # load train params
        train_params = cfg.clf_train_params

        # create data and target vectors
        unknown_map = list()
        unknown_data = np.zeros(shape=(len(unknown_nuclei), len(train_params)))

        # go through nuclei and build target vector
        for i, nucleus in enumerate(unknown_nuclei):
            # get relevant params from nucleus
            for j, key in enumerate(train_params):
                unknown_data[i][j] = nucleus[key]

            # add nucleus to list
            unknown_map.append(nucleus['nID'])

        return unknown_data, unknown_map

    def train_and_predict(self):
        """
        Train the classifier, predict nuclei and
        return probabilities

        :param classifier:
        :return:
        """

        # train the classifier
        self.clf.fit(self.train_data, self.target_data)

        # predict from unkown
        predicted_probas = self.clf.predict_proba(self.unknown_data)

        return predicted_probas

    def eval_prediction(self, predicted_probas, data, target):
        """
        Evaluate performance of classifier

        :param predicted_probas:
        :param data:
        :param target:
        :return:
        """
        # true pos and negative and type I and II errors
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        # go through prediction
        for i, predict in enumerate(predicted_probas):
            if predict[1] > cfg.clf_sig_threshold:
                predict_bool = 1
            else:
                predict_bool = 0

            if target[i] == predict_bool:
                if predict_bool == 1:
                    true_neg += 1
                else:
                    true_pos += 1
            else:
                if predict_bool == 1:
                    false_neg += 1
                else:
                    false_pos += 1

        print('%i:%i:%i:%i' % (true_pos, true_neg, false_pos, false_neg))

        # calculate F1
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * ((precision * recall) / (precision + recall))

        print('F1 = precision: %.2f, recall: %.2f, f1: %.2f' % (precision, recall, f1))
        # or use classifier.score(data, target)
        print('Classifier score = ', self.clf.score(data, target))

        # importance of features
        if hasattr(self.clf, 'feature_importances_'):
            importances = self.clf.feature_importances_

            print('Importance of parameters')
            print(cfg.train_params)
            print(importances)
