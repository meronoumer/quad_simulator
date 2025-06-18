from quad_check import *
import random

# Random vector generator
def make_vectors(k,n,allow_dup = False):
   max_value = 2**n
   if allow_dup ==True:
      sample =  [random.getrandbits(n) for _ in range(k)]
      return sample
   else:
      sample = random.sample(range(0,max_value),k)   
      return sample      



#Making a function that estimates the number of quads in a
def quad_prob(k,n):
    count = 0
    for item in range(1000):
        layout = make_vectors(k,n,allow_dup= False)
        if has_quad_vector(layout):
            count +=1
    quad_prob =  count/ 1000
    return quad_prob



    

    


