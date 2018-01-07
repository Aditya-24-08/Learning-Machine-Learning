import numpy as np
def apply_model(features,is_virginica,model):
	best_acc,best_i,best_t = model
	predictions = np.ones(features.shape[0],bool)
	for example,j in zip(features,range(features.shape[0])):
		if(example[best_i]<=best_t):
			predictions[j] = False
	return predictions
