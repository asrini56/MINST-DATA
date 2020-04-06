# -*- coding: utf-8 -*-
"""
@author: Ashwin Srinivasan
"""
import numpy as np
import pandas as pd
import scipy.io
from math import exp
import math

mat = scipy.io.loadmat('mnist_data.mat')
trX = mat.get('trX')
tsX = mat.get('tsX')
tsY = mat.get('tsY')
trY = mat.get('trY')
trXnew = []
tsXnew = []
trXnew7 = []
trXnew8 = []
X7 = []
X8 = []
s = 0
s1 = 0
s2 = 0
s3 = 0
Co7 = []
Co8 = []

"""
Feature Extraction By calculating mean and standard deviation
"""
for i in range(len(trX)):
        a = np.mean(trX[i])
        b = np.std(trX[i])
        trXnew.append([a,b])

for j in range(len(tsX)):
        a1 = np.mean(tsX[j])
        b1 = np.std(tsX[j])
        tsXnew.append([a1,b1])

print("Feature Extraction:")
print("Length of row for training set before Feature extraction = ",len(trX))
print("Length of coloumn for training set before Feature extraction = ",
      len(trX[0]))
print("Length of row for testing set before Feature extraction = ",len(tsX))
print("Length of coloumn for testing set before Feature extraction = ",
      len(tsX[0]))

print("Length of row for training set after Feature extraction = ",
      len(trXnew))
print("Length of coloumn for training set after Feature extraction = ",
      len(trXnew[0]))
print("Length of row for testing set after Feature extraction = ",len(tsXnew))
print("Length of coloumn for testing set after Feature extraction = ",
      len(tsXnew[0]))

        
"""
Naive Bayes
"""

"""
Seperating the data into 2 sets
trXnew7 represents data for digits 7
trXnew8 represents data for digits 8
"""

print("Naive Byes:")

for i in range(len(trXnew)):
        if trY[0][i] == 0:
            trXnew7.append(trXnew[i])
        else:
            trXnew8.append(trXnew[i])
            
print("Length of row for training set 7 after class seperation = ",
      len(trXnew7))
print("Length of coloumn for training set 7 after class seperation = ",
      len(trXnew7[0]))
print("Length of row for training set 8 after class seperation = ",len(trXnew8))
print("Length of coloumn for training set 8 after class seperation = ",
      len(trXnew8[0]))

            
"""
Prior Probability for digit 7 and digit 8
"""
total_data = 12116
total_data_7 = 6265
total_data_8 = 5851
prior_7 = total_data_7 / total_data
prior_8 = total_data_8 / total_data

            
"""
Finding the mean and covarient matrix for digit 7 and digit 8
"""
for i in range(len(trXnew7)):
    s = s + trXnew7[i][0]
    s1 = s1 + trXnew7[i][1]
b = np.transpose(trXnew7)
cd = np.std(b[0])
cd1 = np.std(b[1])
Co7.append([pow(cd,2),0])
Co7.append([0,pow(cd1,2)]) 
Co7inverse = []
Co7inverse.append([1/pow(cd,2),0])
Co7inverse.append([0,1/pow(cd1,2)])
    
s = s / 6265
s1 = s1 /6265
for i in range(len(trXnew8)):
    s2 = s2 + trXnew8[i][0]
    s3 = s3 + trXnew8[i][1]
s2 = s2 / 5851
s3 = s3 / 5851
c = np.transpose(trXnew8)
cd2 = np.std(c[0])
cd3 = np.std(c[1])
Co8.append([pow(cd2,2),0])
Co8.append([0,pow(cd3,2)]) 
Co8inverse = []
Co8inverse.append([1/pow(cd2,2),0])
Co8inverse.append([0,1/pow(cd3,2)])
    

X7.append(s)
X7.append(s1)
X8.append(s2)
X8.append(s3)
matrix1 = []
for i in range(len(X7)):
    res = X7[i] - Co7[i]
    matrix1.append(res)
matrix1transpose = np.transpose(matrix1)
tsXnewtranspose = np.transpose(tsXnew)

print("Covarient matrix for digit 7 is = ",Co7)
print("Covarient matrix for digit 8 is = ",Co8)

"""
Naive bayes formula for calculating the probability and predicting the class
of the incoming data is done in this function formula()
"""

def naive_bayes(x):
    p = 1/(math.sqrt(2*3.14) * cd)
    e0 = exp((-1/(2*pow(cd,2))) * pow((x[0] - X7[0]),2))
    p1 = 1/(math.sqrt(2*3.14) * cd1)
    e1 = exp((-1/(2*pow(cd1,2))) * pow((x[1] - X7[1]),2))
    result = (p*e0)*(p1*e1)*prior_7
    
    p2 = 1/(math.sqrt(2*3.14) * cd2)
    e2 = exp((-1/(2*pow(cd2,2))) * pow((x[0] - X8[0]),2))
    p3 = 1/(math.sqrt(2*3.14) * cd3)
    e3 = exp((-1/(2*pow(cd3,2))) * pow((x[1] - X8[1]),2))
    result1 = (p2*e2)*(p3*e3)*prior_8
    if result > result1:
        return 0
    else:
        return 1
pred = []

for i in range(len(tsXnew)):
    pre = naive_bayes(tsXnew[i])
    pred.append(pre)

"""
Estimating and printing the accuracy of the model
"""

def predict_naive_bayes_7():
    count = 0
    for i in range(len(pred)):
        if pred[i] == tsY[0][i] and tsY[0][i] == 0:
            count = count + 1
    acc7 = (count/1028)*100
    return acc7


def predict_naive_bayes_8():
    count = 0
    for i in range(len(pred)):
        if pred[i] == tsY[0][i] and tsY[0][i] == 1:
            count = count + 1
    acc7 = (count/974)*100
    return acc7


def predict_naive_bayes():
    count = 0
    for i in range(len(pred)):
        if pred[i] == tsY[0][i]:
            count = count + 1
    acc = (count/2002)*100
    return acc

accuracy_naive_bayes_7 = predict_naive_bayes_7() 
accuracy_naive_bayes_8 = predict_naive_bayes_8()
accuracy_naive_bayes = predict_naive_bayes() 

print("Prior Probability for digit 7 = ",prior_7)
print("Prior Probability for digit 8 = ",prior_8) 
print("Accuracy for digit 7 using naive bayes =",accuracy_naive_bayes_7) 
print("Accuracy for digit 8 using naive bayes =",accuracy_naive_bayes_8)
print("Accuracy for digit 7 and digit 8 using naive bayes ="
      ,accuracy_naive_bayes)


"""
Logistic Regression
"""

trmean = []
trstd = []
trlabel = []
tsmean = []
tsstd = []
tslabel = []

"""
Sigmoid Function for logistic Regression
"""

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

"""
Forming the training and testing data for logistic regression based on the
data obtained from feature extraction [from tsXnew,tsY,trXnew,trY]
"""

def extract(trXnew,tsXnew,trY,tsY):
    for i in range(len(trXnew)):
        trmean.append(trXnew[i][0])
        trstd.append(trXnew[i][1])
    
    for i in range(len(tsXnew)):
        tsmean.append(tsXnew[i][0])
        tsstd.append(tsXnew[i][1])
    
    for i in range(len(trY[0])):
        trlabel.append(trY[0][i])
    
    for i in range(len(tsY[0])):
        tslabel.append(tsY[0][i])
    
    tr_data = {
        "x1" : trmean,
        "x2" : trstd,
        "y" : trlabel
    }
    
    ts_data = {
        "x1" : tsmean,
        "x2" : tsstd,
        "y" : tslabel
    }
    
    training_data = pd.DataFrame(tr_data)
    testing_data = pd.DataFrame(ts_data)
    return training_data,testing_data,trlabel

"""
    Parameter estimation using gradient ascent
"""
def gradient_ascent(X,W,m,n,alpha,iterations,trlabel):
    for i in range(iterations):
        thetaX = np.matmul(X, W)
        var = []
        for x in thetaX:
            var.append(sigmoid(x))
        gradient = np.dot(np.subtract(trlabel, var), X)
        W = np.add(W, (alpha * 1.0/m) * gradient)
    return W

"""
Estimation of the probability and classification of the incoming data and
predicting the accuracy of the model is done in the below steps
"""

def predictlg(t_X,W):
    thetaX = np.matmul(t_X, W)
    test_h = sigmoid(thetaX)
    predict_Y = []
    np.where(test_h >= 0.5, 1., 0.)
    for i in range(len(test_h)):
        if test_h[i] >= 0.5:
            predict_Y.append(1.0)
        else:
            predict_Y.append(0.0)
    return predict_Y

def check_accuracylg(predict_Y,testing_data,t_m):
    accuracy = np.count_nonzero(predict_Y == 
                                testing_data['y'].to_numpy()) * 1.0 / t_m
    return accuracy

def logistic_regression(alpha=4, iterations=100, decision_boundary=0.5):
    training_data,testing_data,trlabel = extract(trXnew,tsXnew,trY,tsY)
    n = 2
    m = training_data['y'].count()
    W = np.zeros(3)
    x1 = training_data['x1']
    x2 = training_data['x2']
    X = np.column_stack((np.ones((m, 1)), x1, x2))
    alpha = 4; iterations = 100
    W = gradient_ascent(X,W,m,n,alpha,iterations,trlabel)
    t_X1 = testing_data['x1']
    t_X2 = testing_data['x2']
    t_mat = len(t_X1)
    t_X = np.column_stack((np.ones((t_mat, 1)), t_X1, t_X2))
    predict_Y = predictlg(t_X,W)
    accuracy_lg = check_accuracylg(predict_Y,testing_data,t_mat)
    return accuracy_lg
    
lg = logistic_regression()
print("Logistic Regression:")
print("Accuracy for the digits using logistic regression = ",lg*100)