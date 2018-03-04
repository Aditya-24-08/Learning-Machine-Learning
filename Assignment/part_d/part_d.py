from csv_import import *
import matplotlib.pyplot as plt
#Normalizing features except x0
for i in [1, 2, 3, 4]:
    X[:, i] = (X[:, i] - X[:, i].mean())/(X[:, i].std())
    
def main(alpha=None, num=None):
    if alpha is None:
        alpha = 0.05
    if num is None:
        num = 10
    print("Problem statement:")
    print("Part (d): experimenting with cost functions" )
    
    print("-\tUse linear combination of the features.\n\
-\tSolve the problem by minimizing different cost functions: \
(i) mean absolute error, (ii)mean squared error, and (iii) mean cubed error. Do not use any regularization.\n\
-     Use gradient descent to minimize the cost function in each case.\n\
-	Plot the test RMSE vs learning rate for each of the cost functions. Which one would you prefer for this problem and why?")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("\nSolution:\n")
    print("Gradient descent will continue iterating \
for ",num," iterations")
    print("If the alpha provided to gradient descent function leads to \
divergence, then an error will be displayed otherwise the learned \
values of the parameters will be printed and the graph between RMSE and number of \
iterations will be plotted")
    print("(i) Mean Absolute Error")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, num)
    print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    print("(ii) Mean squared error")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, num, 2)

    print("\nLearned values of parameters:\n ", theta)
        
    wait = input("PRESS ENTER TO CONTINUE.")
    
    print("(i) Mean Cubed Error")
    wait = input("PRESS ENTER TO CONTINUE.")
    print("Running gradient descent... alpha = ", alpha)
    theta = grad_des(alpha, num, 3)
  
    print("\nLearned values of parameters:\n ", theta)
    
def grad_des(alpha, num_iterations, choice=None):
    if choice is None:
        choice = 1
    theta = np.random.random(5).reshape(5, 1)
    theta*=1000
    m = X.shape[0]
    rmse = []
    i=0
    if(choice==1):
        while (i<num_iterations):
            theta = theta - alpha*np.dot((X.transpose()), np.sign(np.dot(X, theta) - y))/m
            rmse.append(test_err(predict(theta)))
            i+=1
    if(choice==2):
        while (i<num_iterations):
            theta = theta - alpha*np.dot((X.transpose()), (np.dot(X, theta) - y))/m
            rmse.append(test_err(predict(theta)))
            i+=1
    if(choice==3):
        while (i<num_iterations):
          #print(np.dot(X, theta) - y)
            theta = theta - alpha*np.dot((X.transpose()), np.power((np.dot(X, theta) - y), 2))/m
            rmse.append(test_err(predict(theta)))
            i+=1
   # print(num_iterations,  len(rmse))
    rmse = np.array(rmse)
    Plot_RMSE_vs_numiter(rmse, num_iterations, choice)
    return theta

def cost1(X, y, theta):
    hypothesis = np.dot(X, theta)
    cost = np.sum(np.fabs(hypothesis - y))/X.shape[0]
    return cost

def cost2(X, y, theta):
    hypothesis = np.dot(X, theta)
    cost = np.sum(np.power((hypothesis - y), 2))/(2*X.shape[0])
    return cost

def cost3(X, y, theta):
    hypothesis = np.dot(X, theta)
    cost = np.sum(np.power((hypothesis - y), 3))/(3*X.shape[0])
    return cost

def predict(theta):
    for i in [1, 2, 3, 4]:
        Xtest[:, i] = (Xtest[:, i] - Xtest[:, i].mean())/(Xtest[:, i].std())
    predictions = np.dot(Xtest, theta)
    return predictions
    
def test_err(predictions):
    N = ytest.shape[0]
    #print(predictions - ytest)
    sq_diff = (np.power((predictions - ytest), 2))/N
    rmse = np.power(np.sum(sq_diff), 0.5)
    #print("Root mean squared error is:", rmse)
    return rmse
  
def Plot_RMSE_vs_numiter(rmse, num_iterations, choice=None):
    if choice is None:
        choice = 1
    plt.scatter(range(1, num_iterations+1),rmse)
    plt.xlabel("Number of iterations")
    plt.ylabel("RMSE")
    if choice is 1:
        plt.title("Plot of RMSE v/s no. of iterations when cost function is mean absolute error")
    else:
        if choice is 2:
            plt.title("Plot of RMSE v/s no. of iterations when cost function is mean squared error")
        else:
            plt.title("Plot of RMSE v/s no. of iterations when cost function is mean cubed error")

    plt.show()  
