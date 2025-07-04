from functools import reduce,lru_cache
from operator import xor
import random
import itertools 
import numpy as np
from numba import jit,int64,uint64

#setting up the card object 

class Card:
    numbers  = {
        "1":"00",
        "2":"01",
        "3":"10",
        "4":"11"
    }
    colors = {
        "red":"00",
        "green":"01",
        "blue":"10",
        "yellow":"11"
    }
    shapes = {
        "circle":"00",
        "triangle": "01",
        "square": "10",
        "heart" :"11"

    }

    def __init__(self,number,color,shape):
        self.number_value = number
        self.color_value = color
        self.shape_value = shape

    #Getting the binary representations
        binary_num_str = Card.numbers[number]
        binary_color_str = Card.colors[color]
        binary_shape_str = Card.shapes[shape]
    #Converting from binary back to integer form
        int_number = int(binary_num_str, 2)
        int_color = int(binary_color_str, 2)
        int_shape = int(binary_shape_str, 2)  
    #collecting our binary representations of our attributes 
    #and concatenating into 6 digit representation

    #usingbitwise operation shift for all our attributes
        shifted_shape = int_shape << 4
        shifted_color = int_color << 2
    #combining by bitwise OR
        self.vector_representation = shifted_shape | shifted_color | int_number
    
    def __repr__(self):
        return f"This card is a {self.number_value},{self.color_value},{self.shape_value}. The Binary representation is {bin(self.vector_representation)}"

#Randomly generating vectors
def make_vectors(k,n,allow_dup = False):
   if allow_dup:
      sample =  [random.getrandbits(n) for _ in range(k)]
      return sample
   else:
      if 2**n<k:
        raise ValueError("Cannot generate k unique vectors with 2^n < k")
      result = set()
      while len(result) < k:
        result.add(random.getrandbits(n))
      return list(result)  

#Checks for quads
def has_quad(layout):
    # form all possible combinations of 4 set from the layout
    
        #then get the vector_representation of all the cards 
    for commbination in itertools.combinations(layout,4):
     vector_representation = []
     for card in commbination:
        vector_representation.append(card.vector_representation)
        bitwise_xor = reduce(xor, vector_representation)

        if bitwise_xor == 0:
            # print(f'Quad status {True}' )
            return True

    print(f'Quad status {False}' )
    return False



quad_cache = {}

def has_quad_vector_cached(vec_list):
    key = tuple(sorted(vec_list))

    if key in quad_cache:
        return quad_cache[key]

    # If not in cache, compute
    vec_array = np.array(vec_list, dtype=np.int64)
    result = has_quad_vector_numba(vec_array) == 1

    # Store in cache
    quad_cache[key] = result
    return result

@jit(int64(uint64[:]),nopython = True,cache = True)
def has_quad_vector_numba(vec_np_array):
   
   if len(vec_np_array) <4:
      return 0
   for i in range(len(vec_np_array)):
      for j in range(i + 1,len(vec_np_array)):
         for k in range(j+1,len(vec_np_array)):
            for l in range(k +1,len(vec_np_array)):
               if (vec_np_array[i]^vec_np_array[j]^vec_np_array[k]^vec_np_array[l] ==0 ):
                  return 1
   return 0

              
   
def has_quad_vector(vec_list):
    vec_array = np.array(vec_list,dtype=np.int64)
    return has_quad_vector_numba(vec_array)==1
 