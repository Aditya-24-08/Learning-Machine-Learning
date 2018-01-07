from fetch_data import *
from learn_seed import *
import numpy as np
labels = learn_seed(data)
features = data[:,range(data.shape[1] - 1)]
labels = (np.array)(labels)
features = (np.array)(features)
data = (np.array)(features)