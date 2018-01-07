
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
def decide(m,n):
	if(m==0):
		xlabel = 'Sepal length (cm)'
	elif(m==1):
			xlabel = 'Sepal width (cm)'
	elif(m==2):
		xlabel = 'Petal length (cm)'
	elif(m==3):
		xlabel = 'Petal width (cm)'
	else:
		xlabel = 'not defined'

	if(n==0):
		ylabel = 'Sepal length (cm)'
	elif(n==1):
			ylabel = 'Sepal width (cm)'
	elif(n==2):
		ylabel = 'Petal length (cm)'
	elif(n==3):
		ylabel = 'Petal width (cm)'
	else:
		ylabel = 'not defined'	
	return xlabel,ylabel


def plot_data(m,n):

	xlabel,ylabel = decide(m,n)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for t,marker,c in zip(range(3),">ox","rgb"):
		plt.scatter(features[target==t,m],features[target==t,n],marker=marker,c=c)

	plt.legend(["Iris Setosa","Iris Versicolor","Iris Virginica"])	
