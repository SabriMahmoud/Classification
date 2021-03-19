import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data=pd.read_csv('ex2data1.txt',names=['Exam1','Exam2','Admitted'])
data[['Exam1','Exam2']]=(data[['Exam1','Exam2']]-data[['Exam1','Exam2']].mean())/data[['Exam1','Exam2']].std()
data.insert(0,'Ones',1)
#Selection data  based on  h(theta) [y]  values with the function isin([value])
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]

#Plotting positive data 

figure,ax=plt.subplots(figsize=(20,10))
ax.scatter(positive['Exam1'],positive['Exam2'],s=200,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam1'],negative['Exam2'],s=200,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1_values')
ax.set_ylabel('Exam2_values')

#Preparing Classification functions

#Hypothesis 
def segmoid(z):
    return(1/(1+np.exp(-z)))

#plotting sigmoid Function 
fig,ax=plt.subplots(figsize=(20,10))
nums=np.arange(-10,10,step=1)
ax.plot(nums,segmoid(nums),'r')
##############################
################################
#################################
def costClassification(theta,X,y):
    h=segmoid(X.dot(theta))
    logh=np.log(h)
    log1h=np.log(1-h)
    term1=y*logh
    term2=(1-y)*log1h
    return(-np.sum(term1+term2)/(len(X)))

##################################
#################################
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values.reshape(-1,1)
theta=np.zeros((X.shape[1],1))

cost=costClassification(theta,X,y)

#################################
#################################
#
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = segmoid(np.dot(x, theta))
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost =costClassification(theta,x,y)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta
theta_hist=gradientDescent(X,y,theta,0.03,3,850)
theta=np.zeros((X.shape[1],1))

#Finding theta values that produces the minimum cost by scipy library
import scipy.optimize as sco
#Note on flatten() function: Unfortunately scipy’s fmin_tnc doesn’t work well with column or row vector.
# It expects the parameters to be in an array format. 
#The flatten() function reduces a column or row vector into array format.
result=sco.fmin_tnc(func=costClassification,x0=theta.flatten(),fprime=gradientDescent(X,y,theta,0.03,3,850),args=(X,y.flatten()))

costAfterOpt=costClassification(result[0].reshape(3,1),X,y)
