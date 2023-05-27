import pandas as pd # data process aka read csv file
import numpy as np # lin alg
import matplotlib.pyplot as plt # graph visual
"""
sep: It stands for separator, default is ‘, ‘ as in CSV(comma separated values).
header: It accepts int, a list of int, row numbers to use as the column names, and the start of the data. If no names are passed, i.e., header=None, then,  it will display the first column as 0, the second as 1, and so on.
usecols: It is used to retrieve only selected columns from the CSV file.
nrows: It means a number of rows to be displayed from the dataset.
index_col: If None, there are no index numbers displayed along with records.  
skiprows: Skips passed rows in the new data frame.
"""
def calculateCost(w,b):
    total_Cost = 0
    m = len(X_Train)
    f_wb = w * X_Train + b
    total_Cost = np.sum((f_wb-Y_Train)**2)
    return total_Cost/(2*m)

def derivationCalculate(isW,w,b): # w == true b == false
    total_Cost =0
    f_wb = w * X_Train + b
    if isW:
        total_Cost = np.sum((f_wb-Y_Train)*X_Train)
    else:
        total_Cost = np.sum((f_wb-Y_Train))
    return total_Cost/len(X_Train)
            
def gradientDescent(w,b,alpha):
    curW, curB, prevCost = w, b, calculateCost(w, b)
    for _ in range(1000):
        derivationB = derivationCalculate(False, curW, curB)
        derivationW = derivationCalculate(True, curW, curB)
        curW -= alpha * derivationW
        curB -= alpha * derivationB
        cost = calculateCost(curW, curB)
        change = prevCost-cost
        prevCost = cost
        if change < 0.001:
            break
    return curW, curB

split_ratio = .8
learningRate = .001
w,b = 2,2
hold = pd.read_csv('student_scores.csv',usecols = ["Hours","Scores"]) 
split_index = int(split_ratio * len(hold))
X_Train = hold.get("Hours")[:split_index]
Y_Train = hold.get("Scores")[:split_index]
w,b = gradientDescent(w,b,learningRate)
plt.plot(X_Train,Y_Train,'ro') # third parameter to make dot plot
plt.plot(X_Train,X_Train * w + b)
plt.show()
