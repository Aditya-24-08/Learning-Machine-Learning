from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import numpy as np 
from learn_model import *
from apply_model import *

data = load_iris()
features = data['data']
target = data['target']
target_names = data['target_names']
labels = []

for i in target:
	labels.append(target_names[i])

#selecting only the non-setosa features

labels = np.array(labels)
is_setosa = (labels=='setosa')
features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels=='virginica')

err = 0.0

R = features.shape[0]
l = (np.random.permutation((range(features.shape[0]))))
feat = features[l]
Q = R//5
t1 = feat[range(Q),:]
t2 = feat[range(Q,2*Q),:]
t3 = feat[range(2*Q,3*Q),:]
t4 = feat[range(3*Q,4*Q),:]
t5 = feat[range(4*Q,R),:]
T = (t1,t2,t3,t4,t5)
features = feat
is_virginica = is_virginica[l]
for j in range(len(T)):
	training = np.ones(features.shape[0],bool)
	if(j!=(len(T)-1)):
		training[range(j*Q,j*Q+Q)] = False
	else:
		training[range(j*Q,R)] = False
	testing = ~training
	model = learn_model(features[training],is_virginica[training])
	predictions = apply_model(features[testing],is_virginica[testing],model)
	err+=np.sum(predictions!=is_virginica[testing])
	
temp = (err/features.shape[0])*100
error = "%f percent"%temp

