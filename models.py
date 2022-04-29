import numpy as np
from config import args
import matplotlib.pyplot as plt
from utils import plot_cost,plot_cost_one
import optimizers as optim
from data import data

class GradientDescentLinearRegression:
    def __init__(self,w,learning_rate=0.001, max_iterations=2000, eps=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps
        
    def predict(self, X):
        """Returns predictions array of shape [n_samples,1]"""
        return np.dot(X, self.w.T)
    
    def cost(self, X, y):
        """Returns the value of the cost function as a scalar real number"""
        y_pred = self.predict(X)
        loss = (y - y_pred)**2
        return np.mean(loss)

    def grad(self, X, y):
        """Returns the gradient vector"""
        y_pred = self.predict(X)
        d_intercept = -2*sum(y - y_pred)                   
        d_w = -2*sum(X[:,1:] * (y - y_pred).reshape(-1,1)) 
        g = np.append(np.array(d_intercept), d_w)           # Gradient.
        return g / X.shape[0]                               # Average over training samples.
    
    def fit(self, X, y, method = "standard", verbose = True):
        self.w = np.zeros(X.shape[1])   
        expected_grad = np.array([0,0])
        mt = np.array([0, 0])
        vt = np.array([0, 0])
        velocity = np.array([0, 0])
        self.G = np.zeros(X.shape[1])
                                          # Initialization of params.
        # if method == "adagrad":
        #     self.G = np.zeros(X.shape[1])                 # Initialization of cache for AdaGrad.
        w_hist = [self.w]                                 # History of params.
        cost_hist = [self.cost(X, y)]                     # History of cost.      
        i=0

        for iter in range(self.max_iterations):
            
            g = self.grad(X, y) # Calculate the gradient. 
                                    
            if method == "standard":
                step = self.learning_rate * g 
            elif method == "momentum":
               step = optim.momentum(g,velocity)               
            elif method == "adagrad":
                step = optim.adagrad(g,self.G) 
            elif method == "rmsprop":
                 expected_grad = args.gamma * expected_grad + (1 - args.gamma) * np.square(g)
                 step = optim.rmsprop(g,expected_grad)
            elif method == "adam":
              # approximate first and second moment
                mt = args.beta1 * mt + (1 - args.beta1) * g
                vt = args.beta2 * vt + (1 - args.beta2) * np.square(g) 
                # bias corrected moment estimates
                mhat = mt / (1 - args.beta1 ** (i+1))
                vhat = vt / (1 - args.beta2 ** (i+1))
                step = optim.adam(g,mhat,vhat)                               
            else:
                raise ValueError("Method not supported.")
            self.w = self.w - step                        # Update parameters.
            w_hist.append(self.w)                         # Save to history.
            
            J = self.cost(X, y)                           # Calculate the cost.
            cost_hist.append(J)                           # Save to history.
            i+=1
            if verbose:
                print(f"Iter: {iter}, gradient: {g}, params: {self.w}, cost: {J}")
            
            # Stop if update is small enough.
            if np.linalg.norm(w_hist[-1] - w_hist[-2]) < self.eps:
                break
        
        # Final updates before finishing.
        self.iterations = iter + 1                       
        self.w_hist = w_hist
        self.cost_hist = cost_hist
        self.method = method
        
        return self

# def main(method_name,iterations):
#     print(iterations)
#     X = data.iris.data[:, :2] 
#     y = data.iris.target    
#     Methode_list = ["standard","momentum","adagrad","rmsprop","adam"]
#     if method_name == 'all':
#         list_cost = []
#         for method in Methode_list:
#             model = GradientDescentLinearRegression(args.lr, max_iterations=iterations).fit(X, y, method,True)
#             list_cost.append(model.cost_hist)
#         if (X.shape[1] == 2):
#             plot_cost(list_cost)
#             plt.savefig(args.path + "All_cost_cuve.png")
#             plt.show()
#     else:
#             model = GradientDescentLinearRegression(args.lr, max_iterations=iterations).fit(X, y, method_name,False)       
#             if (X.shape[1] == 2):
#                 plot_cost_one(model,method_name)
#                 plt.savefig(args.path +method_name+ "_cost_cuve.png")
#                 plt.show()               