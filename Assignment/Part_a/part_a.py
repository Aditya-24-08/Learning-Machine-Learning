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
    print("Part (a): implementing linear regression" + "\n-\tUse linear combination of the features.")
    print("-\tMinimize mean squared error cost function (as discussed in class).")
    print("-\tUse  gradient  descent  to  minimize  the  cost  function  (as  discussed  in  class).  Use learning rate of 0.05.")
    print("-\tSolve the problem with and without regularization. \
Show how the test RMSE varies with the weightage of the regularization terms (use same weightage for all features).")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("\nSolution:\n")
    print("Gradient descent will continue iterating \
while the difference of consecutive costs will be greater than ", threshold)
    print("If the alpha provided to gradient descent function leads to \
divergence, then an error will be displayed otherwise the learned \
values of the parameters will be printed and the graph between cost function and number of \
iterations will be plotted")
    print("First solving the problem without regularization...")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, threshold)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nReduce alpha, values are diverging\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Now solving the problem with regularisation...")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Let us \
first plot RMSE with different values of lambda")
    la = Plot_lambdavsRmse();
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha, " and lambda = ", la)
    theta = grad_des_reg(alpha, threshold, la)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nCost was increasing in some iterations when it was expected that it had to\
decrease\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
    
def grad_des(alpha, threshold):
    theta = np.random.random(5).reshape(5, 1)
    theta*=1000
    m = X.shape[0]
    Cost = []
    PreviousCost = cost(X, y, theta) + threshold*10
    CurrCost =cost(X, y, theta)
    num_iterations = 0
    while (PreviousCost - CurrCost >threshold):
        PreviousCost = CurrCost
        theta = theta - alpha*np.dot((X.transpose()), (np.dot(X, theta) - y))/m
        CurrCost = cost(X, y, theta)
        Cost.append(CurrCost)
        num_iterations+=1
    
    if(PreviousCost >= CurrCost):
        Cost = np.array(Cost)
        Plot_cost(Cost, num_iterations)
        return theta
    else:
        return -1

def grad_des_reg(alpha, threshold,  Lambda):
    theta = np.random.random(5).reshape(5, 1)
    theta*=1000
    m = X.shape[0]
    
    if threshold is None:
        for i in range(300):
            theta = theta*(1-alpha*Lambda/m) - alpha*np.dot((X.transpose()), (np.dot(X, theta) - y))/m
            theta[0] += theta[0]*(alpha*Lambda/m)
        return theta
    else:
        Cost = []
        num_iterations = 0
        PreviousCost = cost(X, y, theta) + threshold*10
        CurrCost = cost(X, y, theta)
        while (PreviousCost - CurrCost >threshold):
            PreviousCost = CurrCost
            theta = theta*(1-alpha*Lambda/m) - alpha*np.dot((X.transpose()), (np.dot(X, theta) - y))/m
            theta[0] += theta[0]*(alpha*Lambda/m)
            CurrCost = cost(X, y, theta)
            Cost.append(CurrCost)
            num_iterations+=1
        if(PreviousCost >= CurrCost):
            Cost = np.array(Cost)
            Plot_cost(Cost, num_iterations, 1)
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
    
def Plot_cost(Cost, n, choice=None):
    if choice is None:
        choice = 0
    plt.scatter(range(1, n+1),Cost)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    if choice is 0:
        plt.title("Plot of Cost v/s no. of iterations when regularisation is not used")
    else:
        plt.title("Plot of Cost v/s no. of iterations when regularisation is used")
    plt.show()
        
def Plot_lambdavsRmse(alpha=None, delta=None):
    if alpha is None:
        alpha = 0.05
    if delta is None:
        delta = 0.01
    Lambda = 0
    l = []
    r = []
    for i in range(50):
        rmse = test_err(predict(grad_des_reg(alpha, None, Lambda)))
        l.append(Lambda)
        r.append(rmse)
        Lambda+=delta;
    
    plt.scatter(l, r)
    plt.xlabel("Values of lambda")
    plt.ylabel("Values of rmse")
    plt.title("Plot of RMSE v/s values of lambda")
    plt.show()
    l = np.array(l)
    r = np.array(r)
    lm = l[np.argmin(r)]
    print("The value of lambda for which RMSE is minimum is: ", lm," and the value \
of corresponding RMSE is ",r.min() )
    return lm
