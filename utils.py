import matplotlib.pyplot as plt

def plot_cost(list_cost): # this is to plot al the cuve of our optimizer
    """2D plot of the cost on each iteration."""
    fig, ax = plt.subplots(figsize = (4, 3), dpi = 300)
    ax.plot(list_cost[0],color = "blue", label = "GD")
    ax.plot(list_cost[1],color = "red", label = "momentum")
    ax.plot(list_cost[2],color = "green", label = "adagrad")
    ax.plot(list_cost[3],color = "yellow", label = "rmsprop")
    ax.plot(list_cost[4],color = "black", label = "adam")
    ax.legend()                #order["standard","momentum","adagrad","rmsprop","adam"]
    plt.xlabel("iteration")
    plt.ylabel("$J(\mathbf{w})$")
                                

def plot_cost_one(model,methode):  # This is to plot a single cuve
    fig, ax = plt.subplots(figsize = (4, 3), dpi = 300)
    ax.plot(model.cost_hist,color = "blue", label = methode)
    ax.legend()
    plt.xlabel("iteration")
    plt.ylabel("$J(\mathbf{w})$")                                                