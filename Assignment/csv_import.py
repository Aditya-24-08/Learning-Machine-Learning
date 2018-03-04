import csv
import numpy as np
name = "data.csv"

fields = []
Xdata = []
ydata = []
with open(name, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    
    for row in csvreader:
        row = list(map(float, row))
        temp = row[0:4]
        temp = [1] + temp
        Xdata.append(temp)
        ydata.append(row[4])

Xdata = np.array(Xdata)
ydata = np.array(ydata)

no_rows = Xdata.shape[0]
train_rownum = (int)(no_rows*0.8)

X = Xdata[:train_rownum].copy()
y = ydata[:train_rownum].copy()
y = y.reshape(y.shape[0], 1)
Xtest = Xdata[train_rownum:].copy()
ytest = ydata[train_rownum:].copy()
ytest = ytest.reshape(ytest.shape[0], 1)



