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

for j in range(features.shape[0]):
	training = np.ones(features.shape[0],bool)
	training[j] = False
	testing = ~training
	model = learn_model(features[training],is_virginica[training])
	predictions = apply_model(features[testing],is_virginica[testing],model)
	err+=np.sum(predictions!=is_virginica[testing])
	
temp = (err/features.shape[0])*100
error = "%f percent"%temp