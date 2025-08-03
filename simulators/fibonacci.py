from quad_init import make_vectors, has_quad_vector,has_quad_vector_numba
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import time
import tqdm


def has_quad_vector(vec_list):
    vec_array = np.array(vec_list, dtype=np.uint64)
    return has_quad_vector_numba(vec_array) == 1

def one_trial(args):
    k, n = args
    layout = make_vectors(k, n, allow_dup=False)
    layout = np.array(layout, dtype=np.uint64)
    return has_quad_vector(layout)

def quad_probability_parallel(k, n, num_trials=10000):
    list_of_trial_args = [(k, n)] * num_trials
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(one_trial, list_of_trial_args)
    total_quads = sum(results)
    return total_quads / num_trials

def fibonacci_mod2_vectors(k, n):
    """Generate k vectors of dimension n using the Fibonacci sequence mod 2."""
    fib = [1, 1]
    while len(fib) < k * n:
        fib.append((fib[-1] + fib[-2]) % 2)
    vectors = []
    for i in range(k):
        chunk = fib[i * n : (i + 1) * n]
        val = int("".join(str(b) for b in chunk), 2)
        vectors.append(val)
    return vectors

def one_trial_fibonacci(k, n):
    layout = fibonacci_mod2_vectors(k, n)
    layout = np.array(layout, dtype=np.uint64)
    return has_quad_vector(layout)

def compare_random_vs_fibonacci(n, k_values, num_trials=1000):
    probs_random = []
    probs_fibo = []

    for k in tqdm.tqdm(k_values, desc=f"Running comparisons for n={n}"):
        prob_rand = quad_probability_parallel(k, n, num_trials=num_trials)
        probs_random.append(prob_rand)

        prob_fibo = float(one_trial_fibonacci(k, n))  # Only one layout since deterministic
        probs_fibo.append(prob_fibo)

    return probs_random, probs_fibo

def plot_comparison(k_values, probs_random, probs_fibo, n):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, probs_random, label="Random", marker='o')
    plt.plot(k_values, probs_fibo, label="Fibonacci mod 2", marker='s')
    plt.xlabel("Layout size k")
    plt.ylabel("Probability of at least one quad")
    plt.title(f"Quad Probability in F₂^{n}: Random vs. Fibonacci mod 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_comparison_dataframe(n, k_values, probs_random, probs_fibo):
    data = {
        "n": [n] * len(k_values),
        "k": k_values,
        "random_quad_probability": probs_random,
        "fibonacci_mod2_quad_probability": probs_fibo,
        "difference": [r - f for r, f in zip(probs_random, probs_fibo)]
    }
    return pd.DataFrame(data)

# Simulation loop
if __name__ == '__main__':
    start = time.time()

    n = 11
    k_values = list(range(4, 65, 4))  # Example: [4, 8, 12, ..., 64]
    probs_random, probs_fibo = compare_random_vs_fibonacci(n, k_values, num_trials=1000)
    plot_comparison(k_values, probs_random, probs_fibo, n)
    df = build_comparison_dataframe(n, k_values, probs_random, probs_fibo)
    end = time.time()
    df.columns = [f'n={col_name}' for col_name in df.columns]
    df.to_csv(f'output.csv')
    print(f'Time taken for the simulation: {end - start:.2f} seconds')