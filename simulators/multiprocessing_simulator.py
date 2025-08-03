import matplotlib.pyplot as plt
from quad_init import make_vectors,has_quad_vector_numba as has_quad_vector,has_quad_vector_cached
import multiprocessing
import itertools
import time
import numba
import numpy as np
from line_profiler import profile
import pandas as pd
import tqdm


def one_trial(args):
    k,n = args
    layout = make_vectors(k, n, allow_dup=False)
    #converting my layout into a numpy array so it can be sped up with numba
    layout = np.array(layout, dtype=np.uint64)
    if has_quad_vector(layout):
        return 1
    else:
        return 0

def quad_probability_parallel(k,n,num_trials = 10000):
    list_of_trial_args  = []
    for i in range(num_trials):
        list_of_trial_args.append((k,n))

    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        booleans_parallel = pool.map(one_trial,list_of_trial_args)
    
    total_quads = sum(booleans_parallel)
    # print(booleans_parallel)
    # print(total_quads)

    prob = total_quads/num_trials
    
    return prob



# Simulation loop
if __name__ == '__main__':
  k_values = range(8, 24)  # Larger layout sizes
  n_values = [8,9,10,11]   #Deck Sizes = 2^n
  comb_and_prob = {}
  start = time.time()
  for n in n_values:
    comb_and_prob[n] = []
    for k in k_values:
        if k >2**n:
            comb_and_prob[n].append('Nan')
            continue
        prob = quad_probability_parallel(k,n,num_trials = 1000)
        print(f'{k},{2**n}  Probability = {prob}')
        comb_and_prob[n].append(prob)

  end = time.time()
  df = pd.DataFrame(comb_and_prob, index=list(k_values))
  df.columns = [f'n={col_name}' for col_name in df.columns]
  df.to_csv("quad_probabilities.csv")
  print(f'Time taken for simulation = {end - start}')

  for n, prob in comb_and_prob.items():
        plt.plot(list(k_values), prob, label=f'n = {n} ($2^{{{n}}}$ vectors)', marker='o', markersize=4, linestyle='-')


    # Visualizing 
  plt.xlabel(f"Layout size {2**k} ")
  plt.ylabel("Probabilities of finding a quad")
  plt.title('Probability of finding a quad in different Layouts\n using the monte Carlo Simulation ')

  plt.grid(True) 
  plt.xlim(k_values.start, k_values.stop - 1)
  plt.ylim(0, 1)
  plt.tight_layout()
  plt.legend()
  plt.show()