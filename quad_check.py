from functools import reduce
from operator import xor
import random
import itertools 
import numpy as np

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


#making a function to check if a layout contains a quad 
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


def has_quad_vector(vec_list):
    for combo in itertools.combinations(vec_list, 4):
        if reduce(xor, combo) == 0:
            return True
    return False
