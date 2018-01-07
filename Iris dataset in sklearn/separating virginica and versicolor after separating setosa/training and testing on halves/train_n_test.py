#training on 50% data and testing on the other 50%
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

rows = features.shape[0]
half = rows//2
#denoting training by 1 and testing by 2
features1 = features[:half,:]
features2 = features[half:,:]
is_virginica1 = is_virginica[:half]
is_virginica2 = is_virginica[half:]
best_acc = -1
for i in range(features1.shape[1]):
	thresh = features1[:,i].copy()
	thresh.sort()
	for t in thresh:
		pred = (features1[:,i] > t)
		acc = (pred==is_virginica1).mean()
		if acc > best_acc:
			best_acc = acc
			best_i = i
			best_t = t 
		training_acc = acc
		testing_acc = 0 
		for it,v in zip(features2,is_virginica2):
			if(it[i]>t):
				#it is classified as virginica by our trainer
				if(v):
					testing_acc = testing_acc + 1
			else:
				#it is classified as versicolor by our trainer
				if(~v):
					testing_acc = testing_acc + 1

		testing_acc = testing_acc/half
		if(testing_acc>0.9 and training_acc>0.9):
			print(training_acc,i,t,testing_acc)

#test on one half, train on other half and then print those outcomes which have both training and testing accuracies good enough