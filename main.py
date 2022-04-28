import matplotlib.pyplot as plt
import numpy as np
from models import main

# MAX_ITER = 2000  has to be provide by the user
# METHOD = "momentum" has to be provide by the user 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m','--optimizer',help='This is the name of the optimizer',required=True)
parser.add_argument('-n','--max_iter',help='This is the number of iteration',type=int,required=True)
mains_args = vars(parser.parse_args())

main(mains_args['optimizer'],mains_args['max_iter'])