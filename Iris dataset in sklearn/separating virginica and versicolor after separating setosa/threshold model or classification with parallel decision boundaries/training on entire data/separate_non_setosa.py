from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
import numpy as np 

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

best_acc = -1
for i in range(features.shape[1]):
	thresh = features[:,i].copy()
	thresh.sort()
	for t in thresh:
		pred = (features[:,i] > t)
		acc = (pred==is_virginica).mean()
		if acc > best_acc:
			best_acc = acc
			best_i = i
			best_t = t 





