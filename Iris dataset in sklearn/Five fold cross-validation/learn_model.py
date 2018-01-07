import numpy as np
def learn_model(features,is_virginica):
	best_acc = -1
	for i in range(features.shape[1]):
		thresh = features[:,i].copy()
		thresh.sort()
		for t in thresh:
			pred = (features[:,i] > t)
			acc = (pred==is_virginica).mean()
			if(acc > best_acc):
				best_acc = acc
				best_i = i
				best_t = t

	model = (best_acc,best_i,best_t)
	return model