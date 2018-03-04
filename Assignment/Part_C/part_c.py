from csv_import import *
import matplotlib.pyplot as plt
#Normalizing features except x0
for i in [1, 2, 3, 4]:
    X[:, i] = (X[:, i] - X[:, i].mean())/(X[:, i].std())
    
def main(alpha=None,  threshold=None):
    if alpha is None:
        alpha = 0.05
    if threshold is None:
        threshold = 0.0001
    print("Problem statement:")
    print("Part (c): experimenting with combinations of features\n\
-\tMinimize mean squared error cost function. Do not use any regularization.\n\
-\tUse gradient descent to minimize the cost function. Use learning rate of 0.05.\n\
-\tSolve  the  problem  using (i) linear, (ii) quadratic and (iii) cubic combinations of the features.\n\
-\tPlot  the  test  RMSE  vs  learning  rate  for  each of the cases. Which one you would prefer for this problem and why?")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("\nSolution:\n")
    print("Gradient descent will continue iterating \
while the difference of consecutive costs will be greater than ", threshold)
    print("If the alpha provided to gradient descent function leads to \
divergence, then an error will be displayed otherwise the learned \
values of the parameters will be printed and the graph between RMSE and number of \
iterations will be plotted")
    print("First solving the problem using linear combination of features...")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, threshold)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nValues are diverging\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Now solving the problem with quadratic combinations of features...")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, threshold, 2)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nValues are diverging\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
    
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Now solving the problem with cubic combinations of features...")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, threshold, 3)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nValues are diverging\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Now we see that values are diverging for quadratic and cubic combinations \
of features")
    print("So let us plot a graph between RMSE and the learning rate alpha such that\
 the values for which divergence will occur will have infinite RMSE and hence will not\
 be plotted in the graph ")
    print("The learning rate will vary from 0.003 to 0.729 in multiples of three")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("(i) Linear combination of features")
    rmse = []
    alp = []
    print("Plotting RMSE v/s alpha graph...")
    alpha = 0.003
    for i in range(5):
        alp.append(alpha)
        theta = grad_des(alpha, 0.001, None, 1)
        alpha*=3
        try:
            if(theta==-1): #theta=-1  means an error
                rmse.append(np.nan)
        except ValueError:
                rmse.append(test_err(predict(theta)))
    Plot_RMSE_vs_alp(rmse, alp,1)
    
    wait = input("PRESS ENTER TO CONTINUE.")
    print("(ii) Quadratic combination of features")
    rmse = []
    alp = []
    print("Plotting RMSE v/s alpha graph...")
    alpha = 0.003
    for i in range(5):
        alp.append(alpha)
        theta = grad_des(alpha, 0.001, 2, 1)
        alpha*=3
        try:
            if(theta==-1): #theta=-1  means an error
                rmse.append(np.nan)
        except ValueError:
                rmse.append(test_err(predict(theta)))
    Plot_RMSE_vs_alp(rmse, alp,2)
    
    wait = input("PRESS ENTER TO CONTINUE.")
    print("(iii) Cubic combination of features")
    rmse = []
    alp = []
    print("Plotting RMSE v/s alpha graph...")
    alpha = 0.003
    for i in range(5):
        alp.append(alpha)
        theta = grad_des(alpha, 0.001, 3, 1)
        alpha*=3
        try:
            if(theta==-1): #theta=-1  means an error
                rmse.append(np.nan)
        except ValueError:
                rmse.append(test_err(predict(theta)))
    Plot_RMSE_vs_alp(rmse, alp,3)
    print("From the above graphs, we observe that the linear features converge for\
 all alpha values, the quadratic features only converge for alpha=0.003 out of the given\
 values and the cubic features don't converge for any alpha value of the given values")
    print("Since for the given alpha = 0.05, only linear combinational features\
 were leading to convergence, so for this problem, I would choose linear combinations.")
    
def grad_des(alpha, threshold, choice = None, notplot=None):
    if choice is None:
        choice = 1
        Xtrain = X
    else:
        if(choice==2):
            Xtrain = np.power(X, 2)
        else:
            Xtrain = np.power(X, 3)
    if notplot is None:
        notplot = 0
    theta = np.random.random(5).reshape(5, 1)
    theta*=1000
    m = X.shape[0]
    PreviousCost = cost(Xtrain, y, theta) + threshold*10
    CurrCost =cost(Xtrain, y, theta)
    num_iterations = 0
    rmse = []
    while (PreviousCost - CurrCost >threshold):
        PreviousCost = CurrCost
        theta = theta - alpha*np.dot((Xtrain.transpose()), (np.dot(Xtrain, theta) - y))/m
        CurrCost = cost(Xtrain, y, theta)
        num_iterations+=1
        if(notplot==0):
            rmse.append(test_err(predict(theta)))
    
    if(PreviousCost >= CurrCost):
        if(notplot==0):
            Plot_RMSE_vs_numiter(rmse, num_iterations, choice)
        return theta
    else:
        return -1

def cost(X, y, theta):
    hypothesis = np.dot(X, theta)
    cost = np.sum(np.power((hypothesis - y), 2))/(2*X.shape[0])
    return cost

def predict(theta):
    for i in [1, 2, 3, 4]:
        Xtest[:, i] = (Xtest[:, i] - Xtest[:, i].mean())/(Xtest[:, i].std())
    predictions = np.dot(Xtest, theta)
    return predictions
    
def test_err(predictions):
    N = ytest.shape[0]
    sq_diff = (np.power((predictions - ytest), 2))/N
    rmse = np.power(np.sum(sq_diff), 0.5)
    #print("Root mean squared error is:", rmse)
    return rmse
    
def Plot_RMSE_vs_alp(rmse, alp, choice):
    plt.scatter(alp,rmse)
    plt.xlabel("Learning Rate")
    plt.ylabel("RMSE")
    if (choice==1):
        plt.title("Plot of RMSE v/s learning rate when linear combinations of features are used")
    else:
        if(choice==2):
            plt.title("Plot of RMSE v/s learning rate when quadratic combinations of features are used")
        else:
            plt.title("Plot of RMSE v/s learning rate when cubic combinations of features are used")
    plt.show()

def Plot_RMSE_vs_numiter(rmse, num_iterations, choice):
    plt.scatter(range(1, num_iterations+1),rmse)
    plt.xlabel("Number of iterations")
    plt.ylabel("RMSE")
    if (choice==1):
        plt.title("Plot of RMSE v/s no. of iterations when linear combinations of features are used")
    else:
        if(choice==2):
            plt.title("Plot of RMSE v/s no. of iterations when quadratic combinations of features are used")
        else:
            plt.title("Plot of RMSE v/s no. of iterations when cubic combinations of features are used")
    plt.show()
