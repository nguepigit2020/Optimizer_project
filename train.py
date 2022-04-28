# import numpy as np
# import optimizers as optim
# from config import args
# import models


# def fit(X, y, method = "standard", verbose = True):

#         w = np.zeros(X.shape[1])  
#         grad_store = np.zeros([args.max_iterations, 2], dtype=float) 
#         expected_grad = np.array([0,0])
#         mt = np.array([0, 0])
#         vt = np.array([0, 0])
#         velocity = np.array([0, 0])
#                                           # Initialization of params.
#         if method == "adagrad":
#             G = np.zeros(X.shape[1])                 # Initialization of cache for AdaGrad.
#         w_hist = [w]                                 # History of params.
#         cost_hist = [models.cost(X, y,w)]                     # History of cost.      
#         i=0

#         for iter in range(args.max_iterations):
            
#             g = models.grad(X, y,w) # Calculate the gradient.
#             grad_store[i] = g  
                                    
#             if method == "standard":
#                 step = args.lr * g 
#             elif method == "momentum":
#                step = optim.optimizer.momentum(g,velocity)               
#             elif method == "adagrad":
#                 step = optim.optimizer.adagrad(g) 
#             elif method == "rmsprop":
#                  expected_grad = args.gamma * expected_grad + (1 - args.gamma) * np.square(g)
#                  step = optim.optimizer.rmsprop(g,expected_grad)
#             elif method == "adam":
#               # approximate first and second moment
#                 mt = args.beta1 * mt + (1 - args.beta1) * g
#                 vt = args.beta2 * vt + (1 - args.beta2) * np.square(g) 
#                 # bias corrected moment estimates
#                 mhat = mt / (1 - args.beta1 ** (i+1))
#                 vhat = vt / (1 - args.beta2 ** (i+1))
#                 step = optim.optimizer.adam(g,mhat,vhat)                               
#             else:
#                 raise ValueError("Method not supported.")
#             w = w - step                        # Update parameters.
#             w_hist.append(w)                         # Save to history.
            
#             J = models.cost(X, y,w)                           # Calculate the cost.
#             cost_hist.append(J)                           # Save to history.
#             i+=1
#             if verbose:
#                 print(f"Iter: {iter}, gradient: {g}, params: {w}, cost: {J}")
            
#             # Stop if update is small enough.
#             if np.linalg.norm(w_hist[-1] - w_hist[-2]) < args.eps:
#                 break
        
#         # Final updates before finishing.
#         iterations = iter + 1                     
#         w_hist = w_hist
#         cost_hist = cost_hist
#         method = method
        
#         return cost_hist,w