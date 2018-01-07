#plot area v/s compactness

#Features:
#Area
#Perimeter
#Compactness
#Length of kernel
#Width of kernel
#Asymmetry coefficient
#length of kernel groove

import matplotlib.pyplot as plt
import numpy as np
def decide(xlabel,ylabel):
	if(xlabel=='area'):
		a = 0
	elif(xlabel=='perimeter'):
		a = 1
	elif(xlabel=='compactness'):
		a = 2
	elif(xlabel=='length of kernel'):
		a = 3
	elif(xlabel=='width of kernel'):
		a = 4
	elif(xlabel=='asymmetry coefficient'):
		a = 5
	elif(xlabel=='length of kernel groove'):
		a = 6

	if(ylabel=='area'):
		b = 0
	elif(ylabel=='perimeter'):
		b = 1
	elif(ylabel=='compactness'):
		b = 2
	elif(ylabel=='length of kernel'):
		b = 3
	elif(ylabel=='width of kernel'):
		b = 4
	elif(ylabel=='asymmetry coefficient'):
		b = 5
	elif(ylabel=='length of kernel groove'):
		b = 6
	return a,b

def plot_seeds(features,labels,xlabel,ylabel):
	a,b = decide(xlabel,ylabel)
	is_kama = (labels=='Kama')
	is_rosa = (labels=='Rosa')
	is_canadian = (labels=='Canadian')
	H = (is_kama,is_rosa,is_canadian)
	for t,c,m in zip(H,'rgb','x+o'):
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		temp = features[t]
		plt.scatter(temp[:,a],temp[:,b],c=c,marker=m)
		plt.legend(['kama','rosa','canadian'])