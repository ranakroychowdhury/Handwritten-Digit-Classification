# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:53:00 2019

@author: Ranak Roy Chowdhury
"""
import matplotlib
import math
import numpy as np
from numpy import exp
from matplotlib import pyplot
from matplotlib.pyplot import figure


def readFiles():
    matrix1 = np.loadtxt('train3_oddYr.txt')
    matrix2 = np.loadtxt('train5_oddYr.txt')
    three_inst, dim = matrix1.shape
    five_inst, dim = matrix2.shape
    train = np.concatenate((matrix1, matrix2), axis=0)
    
    test3 = np.loadtxt('test3_oddYr.txt')
    test5 = np.loadtxt('test5_oddYr.txt')
        
    return train, three_inst, five_inst, dim, test3, test5


def prepareLabels(three_inst, five_inst):
    three = np.zeros((three_inst, 1))
    five = np.ones((five_inst, 1))
    train_labels = np.concatenate((three, five), axis=0)
    return train_labels


def sigmoid(train_data, weights):
    result = np.matmul(train_data, weights)
    return 1 / (1 + exp(-result))


def prediction(data, weights, target_labels):
    inst, dim = data.shape
    predicted_labels = sigmoid(data, weights)
    predicted_labels = np.where(predicted_labels < 0.5, 0, 1)
    
    error_inst = np.abs(predicted_labels - target_labels)
    error_fraction = np.sum(error_inst)/inst
    
    return error_fraction
    
    
def printTrainingResult(log_likelihood, three_error, five_error):
    print('Log-likelihhod is: ' + str(log_likelihood))
    print('Fraction of error')
    print('Image 3: ' + str(three_error))
    print('Image 5: ' + str(five_error))
    print('\n')
            
    
def learn(train_data, train_labels, three_inst, five_inst, iteration, check):
    inst, dim = train_data.shape
    learning_rate = 0.2/inst
    weights = np.random.rand(dim, 1)
    ones = np.ones((inst, 1))
    likelihood = []
    error_three = []
    error_five = []
    
    for i in range(iteration):
        sigmoid_result = sigmoid(train_data, weights)
        weights += learning_rate * (np.matmul(train_data.transpose(), (train_labels - sigmoid_result)))
        
        if(i % check == 0):
            log_likelihood = np.matmul(train_labels.transpose(), np.log(sigmoid_result)) + np.matmul((ones - train_labels).transpose(), np.log(ones - sigmoid_result))
            three_error = prediction(train_data[0 : three_inst], weights, train_labels[0 : three_inst])
            five_error = prediction(train_data[three_inst : inst], weights, train_labels[three_inst : inst])
            
            # printTrainingResult(log_likelihood[0][0], three_error, five_error)
            likelihood.append(log_likelihood[0][0])
            error_three.append(three_error*100)
            error_five.append(five_error*100)
            
    return weights, likelihood, error_three, error_five
    
    
def test(weights, test3_data, test5_data):
    inst, dim = test3_data.shape
    three_error = prediction(test3_data, weights, np.zeros((inst, 1)))
    print('Error Rate on Image 3\'s test data is: ' + str(three_error*100))
    
    inst, dim = test5_data.shape
    five_error = prediction(test5_data, weights, np.ones((inst, 1)))
    print('Error Rate on Image 5\'s test data is: ' + str(five_error*100))
    
    
def convergencePlot(result, x, string):
    fig = pyplot.gcf()
    fig.set_size_inches(20, 15)
    ax1 = fig.add_subplot(211)
    
    ax1.set_ylabel(string)
    ax1.set_xlabel('Iteration #')
    plt.rcParams.update({'font.size': 22})
    
    pyplot.plot(x, result)
    pyplot.show()
    
    
if __name__ == "__main__":
    print("Reading all files")
    train_data, three_inst, five_inst, dim, test3_data, test5_data = readFiles()
    train_labels = prepareLabels(three_inst, five_inst)
    
    iteration = 1000
    check = 1
    print("Learning parameters of the model")
    weights, likelihood, error_three, error_five = learn(train_data, train_labels, three_inst, five_inst, iteration, check)
    
    x = np.arange(check, iteration + 1, check).tolist()
    print('Plot Results')
    convergencePlot(likelihood, x, 'Log-likelihood')
    convergencePlot(error_three, x, 'Percentage Training Error on Image 3')
    convergencePlot(error_five, x, 'Percentage Training Error on Image 5')
    weight_matrix = weights.reshape(int(math.sqrt(dim)), int(math.sqrt(dim)))
    print(weight_matrix)
    
    print("Performance on Test Data")
    test(weights, test3_data, test5_data)
    print(likelihood[-1])