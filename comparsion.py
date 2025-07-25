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

def thue_morse_bits(length):
    seq = [0]
    while len(seq) < length:
        seq += [1 - b for b in seq]
    return seq[:length]

def thue_morse_vectors(k, n):
    bits = thue_morse_bits(k * n)
    vectors = []
    for i in range(k):
        chunk = bits[i * n : (i + 1) * n]
        # Convert to integer
        val = int("".join(str(b) for b in chunk), 2)
        vectors.append(val)
    return vectors

def one_trial_thue_morse(k, n):
    layout = thue_morse_vectors(k, n)
    layout = np.array(layout, dtype=np.uint64)
    return has_quad_vector(layout)

def compare_random_vs_thuemorse(n, k_values, num_trials=1000):
    probs_random = []
    probs_thue = []

    for k in tqdm.tqdm(k_values, desc=f"Running comparisons for n={n}"):
        # Random
        prob_rand = quad_probability_parallel(k, n, num_trials=num_trials)
        probs_random.append(prob_rand)

        # Thue-Morse (run once; deterministic)
        prob_thue = float(one_trial_thue_morse(k, n))
        probs_thue.append(prob_thue)

    return probs_random, probs_thue

def plot_comparison(k_values, probs_random, probs_thue, n):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, probs_random, label="Random", marker='o')
    plt.plot(k_values, probs_thue, label="Thue–Morse", marker='s')
    plt.xlabel("Layout size k")
    plt.ylabel("Probability of at least one quad")
    plt.title(f"Quad Probability in F₂^{n}: Random vs. Thue–Morse")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd

def build_comparison_dataframe(n, k_values, probs_random, probs_thue):
    data = {
        "n": [n] * len(k_values),
        "k": k_values,
        "random_quad_probability": probs_random,
        "thue_morse_quad_probability": probs_thue,
        "difference": [r - t for r, t in zip(probs_random, probs_thue)]
    }
    df = pd.DataFrame(data)
    return df


# Simulation loop
if __name__ == '__main__':
  start = time.time()
 
  n = 11  # You can try n = 4 to n = 7 or higher
  k_values = list(range(4, 65, 4))  # Step by 4 for speed; adjust as needed
  probs_random, probs_thue = compare_random_vs_thuemorse(n, k_values, num_trials=1000)
  plot_comparison(k_values, probs_random, probs_thue, n)
  df = build_comparison_dataframe(n, k_values, probs_random, probs_thue)
  end = time.time()
  print(f'Time taken for the simulation : {end - start}')

