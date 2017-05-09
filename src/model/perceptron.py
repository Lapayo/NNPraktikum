# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from random import random

from pprint import pprint

from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator


__author__ = "Simon Roesler"  # Adjust this when you copy the file
__email__ = "uzdgn@student.kit.edu"  # Adjust this when you copy the file


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        for i in range(0, self.epochs):
            if (verbose):
                print "\nEpoch:", i
                evaluator = Evaluator()
                evaluator.printAccuracy(self.validationSet, self.evaluate(self.validationSet.input))
                print("\n--------")


            for idx, x in enumerate(self.trainingSet.input):
                t_x = self.trainingSet.label[idx]
                o_x = self.classify(x)

                self.updateWeights(x, t_x - o_x)

        pass

    def classify(self, testInstance):
        return self.fire(testInstance)

    def evaluate(self, test=None):
        if test is None:
            test = self.testSet.input

        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        delta_w = self.learningRate * error * input
        self.weight = self.weight + delta_w
        pass
         
    def fire(self, input):
        return Activation.sign(np.dot(np.array(input), self.weight))
