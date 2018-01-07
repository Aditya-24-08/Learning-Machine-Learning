
def separate_setosa(data):
	import numpy as np
	#We observe that petal length seems to be able to separate Setosa from Versicolor and Virginica
	features = data['data']
	target = data['target']

	plen = features[:, 2] #all rows of the third column which essentially is the feature of petal length
	is_setosa = (target==0)
	max_setosa =  plen[is_setosa].max()
	min_non_setosa = plen[~is_setosa].min()
	Range = [max_setosa, min_non_setosa]
	return Range

def build_model(data,new_unclassified_features):
	#here the max petal length of Iris Setosa is smaller than the min petal length of non setosa, hence we can build a simple model to separate setosa from others as follows
	R = separate_setosa(data)
	
	if(R[0]>=R[1]):
		return -1
	else:
		for i in  (new_unclassified_features[:,2] < 2):
				if(i==True):
					print('Iris Setosa')
				else:
					print('Iris Virginica or Iris Versicolour')

