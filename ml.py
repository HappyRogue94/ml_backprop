from plot import draw_dot

import numpy as np


class Value:
    """
    This class defines a data structure used to represent a mathematical experession
    and its operations 
    """ 
    def __init__(self, data, _children=(), _op='') -> None:
        self.data = data
        self._prev=set(_children)
        self._op=_op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """
        this method is used to overload the '+' operator to allow for addition
        of two seperate objects
        """
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        """
        this method is used to overload the '*' operator to allow for multiplication
        of two seperate objects
        """
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
if __name__ == '__main__':
    
    a = Value(data=2.0)
    b = Value(data=-3.0)
    c = Value(data=10.0)
    d = a*b + c
    draw_dot(d)


