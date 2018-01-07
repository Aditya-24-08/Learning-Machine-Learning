
def learn_seed(data):
	C = data.shape[1]
	target = data[:,C-1]
	labels = []
	for t in target:
		if(t==1):
			g = 'Kama'
		elif(t==2):
			g = 'Rosa'
		else:
			g = 'Canadian'
		labels.append(g)
	return labels