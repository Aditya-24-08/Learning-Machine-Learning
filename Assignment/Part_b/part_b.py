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
    print("Problem Statement\nPart (b): experimenting with optimization algorithms\n\
-\tUse linear combination of the features.\n\
-\tMinimize mean squared error cost function (as discussed in class). Do not use any regularization.\n\
-\tSolve the problem by minimizing the cost function using two optimization algorithms: (i)  gradient  descent  with  learning  rate  of  0.05,  and  (ii)  iterative  re-weighted  least square method. \n\
-\tPlot  the  test  RMSE  vs  number  of  iterations  for  both  the  optimization  algorithms.\n\
Which optimization algorithm would you prefer for this problem and why?")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("\nSolution:\n")
    print("First solving the problem with gradient descent...")
    print("Gradient descent will continue iterating \
while the difference of consecutive costs will be greater than ", threshold)
    print("If the alpha provided to gradient descent function leads to \
divergence, then an error will be displayed otherwise the learned \
values of the parameters will be printed and the graph between RMSE and number of \
iterations will be plotted")
    
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, threshold)
    try:
        if(theta==-1): #theta=-1  means an error
            print("\nReduce alpha, values are diverging\n")
    except ValueError:
        print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    print("\nNow solving the problem with iterative  re-weighted  least square method:")
    print("\n Newton Raphson Update in IRLS for linear regression is independent of theta, so it gives exact solution in one step")
    print("\n Learned Parameters:\n")
    theta2 = IRLS()
    print(theta2)
    print("For the given problem, I would prefer the iterative re-weighted least square method \
because it gives out the values of parameters in just one iteration and that too exactly. \
IRLS is not used in situations where there are million of features to account for \
because inversion of such a big resulting matrix in the algorithm would take a lot of time.")
    
def grad_des(alpha, threshold):
    theta = np.random.random(5).reshape(5, 1)
    theta*=1000
    m = X.shape[0]
    PreviousCost = cost(X, y, theta) + threshold*10
    CurrCost =cost(X, y, theta)
    num_iterations = 0
    rmse = []
    while (PreviousCost - CurrCost >threshold):
        PreviousCost = CurrCost
        theta = theta - alpha*np.dot((X.transpose()), (np.dot(X, theta) - y))/m
        CurrCost = cost(X, y, theta)
        num_iterations+=1
        rmse.append(test_err(predict(theta)))
    rmse = np.array(rmse)
    if(PreviousCost >= CurrCost):
        Plot_RMSE_vs_numiter(rmse, num_iterations)
        return theta
    else:
        return -1

def IRLS():
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    Plot_RMSE_vs_numiter(test_err(predict(theta)), 1, 1)
    return theta
    
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
        
def Plot_RMSE_vs_numiter(rmse, num_iterations, choice=None):
    if choice is None:
        choice = 0
    plt.scatter(range(1, num_iterations+1),rmse)
    plt.xlabel("Number of iterations")
    plt.ylabel("RMSE")
    if choice is 0:
        plt.title("Plot of RMSE v/s no. of iterations when gradient descent is used")
    else:
        plt.title("Plot of RMSE v/s no. of iterations when iterative re-weighted least square is used")
    plt.show()
