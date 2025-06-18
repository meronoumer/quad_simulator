from quad_check import *
import random


def make_vectors(k,n,allow_dup = False):
   max_value = 2**n
   if allow_dup ==True:
      sample =  [random.getrandbits(n) for _ in range(k)]
      return sample
   else:
      sample = random.sample(range(0,max_value),k)   
      return sample      


print(" Sample vectors:", [bin(v) for v in vectors])


def quad_prob(k,n):
    count = 0
    for item in range(1000):
        layout = make_vectors(k,n,allow_dup= False)
        if has_quad_vector(layout):
            count +=1
    quad_prob =  count/ 1000
    return quad_prob

#Testing 
prob = quad_prob(k=10, n=6)
print(f"Estimated probability of a quad in layout: {prob:.4f}")



    

    


