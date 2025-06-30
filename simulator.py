from quad_init import *
import random
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp



#Making a function that estimates the number of quads in a
def quad_prob(k,n):
    count = 0
    num_trials = 1000
    for item in range(num_trials):
        layout = make_vectors(k,n,allow_dup= False)
        if has_quad_vector(layout):
            count +=1
    quad_prob =  count/ num_trials
    return quad_prob

#Making the Monte Carlo simulation 

if __name__ == "__main__":

    k_values = range(4,101)
    n_values = [16,32,64,128,256.]
    results = {}
    """
    for each vector length (n)and each layout size(k) I want to run
    several  trials so i can calculate the probability of a quad appearing in each probability
    for all specific combinations and then just save them onto probabilities

    """

    for n in n_values:
        results[n]=[]
        for k in k_values:
            print("~~~~~~~~~")
            prob = quad_prob(k,n)
            print(f"N =  {n}")
            print(f"K =  {k}")
            results[n].append(prob)

    print(results + "\n done")

    for n, prob in results.items():
        plt.plot(list(k_values),prob)


    # Visualizing 
    plt.plot(results.keys,results.values)
    plt.xlabel("Layout size ,k ")
    plt.ylabel("Probabilities of finding a quad")
    plt.title('Probability of finding a quad in different Layouts')
    plt.grid(True) 
    plt.ylim(0, 1)
    plt.tight_layout() 
    plt.show()
