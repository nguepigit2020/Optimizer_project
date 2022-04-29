from data import data
import matplotlib.pyplot as plt
import numpy as np
# from models import main
from config import args
from models import GradientDescentLinearRegression 
from utils import plot_cost, plot_cost_one
# MAX_ITER = 2000  has to be provide by the user
# METHOD = "momentum" has to be provide by the user 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m','--optimizer',help='This is the name of the optimizer',required=True)
parser.add_argument('-n','--max_iter',help='This is the number of iteration',type=int,required=True)
mains_args = vars(parser.parse_args())
method_name = mains_args['optimizer']
iterations = mains_args['max_iter']
# main(mains_args['optimizer'],mains_args['max_iter'])
# def main(method_name,iterations):

X = data.iris.data[:, :2] 
y = data.iris.target    
Methode_list = ["standard","momentum","adagrad","rmsprop","adam"]
if method_name == 'all':
    list_cost = []
    for method in Methode_list:
        model = GradientDescentLinearRegression(args.lr, max_iterations=iterations).fit(X, y, method,True)
        list_cost.append(model.cost_hist)
    if (X.shape[1] == 2):
        plot_cost(list_cost)
        plt.savefig(args.path + "All_cost_cuve.png")
        plt.show()
else:
        model = GradientDescentLinearRegression(args.lr, max_iterations=iterations).fit(X, y, method_name,False)       
        if (X.shape[1] == 2):
            plot_cost_one(model,method_name)
            plt.savefig(args.path +method_name+ "_cost_cuve.png")
            plt.show()               