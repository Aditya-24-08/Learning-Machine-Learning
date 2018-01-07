import numpy as np
from get_labels import *
def distance(p0,p1):
	return np.sum((p0-p1)**2)

def nn_classify(training_set,training_labels,new_example):
	dists = np.array([distance(t,new_example) for t in training_set])
	nearest = dists.argmin()
	return training_labels[nearest]

#Ten fold cross validation
err = 0.0

R = features.shape[0]
l = (np.random.permutation((range(features.shape[0]))))
feat = features[l]
Q = R//10
T = (list)([])
for i in range(9):
	T.append((np.array)(feat[range(i*Q,i*Q+Q),:]))
T.append((np.array)(feat[range(9*Q,R),:]))
T = (tuple)(T)
features = feat
labels = labels[l]
for j in range(len(T)):
	training = np.ones(features.shape[0],bool)
	if(j!=(len(T)-1)):
		training[range(j*Q,j*Q+Q)] = False
	else:
		training[range(j*Q,R)] = False
	testing = ~training

	testing_feat = features[testing]
	labels_feat = labels[testing]
	for new_example,l in zip(testing_feat,labels_feat):
		prediction = nn_classify(features[training],labels[training],new_example)
		err+=(prediction!=l)
	
temp = (err/features.shape[0])*100
error = "%f percent"%temp

