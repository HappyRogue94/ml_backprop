from plot import draw_dot

import numpy as np
import math


class Value:
    """
    This class defines a data structure used to represent a mathematical experession
    and its operations 
    """ 
    def __init__(self, data, _children=(), _op='', label = '') -> None:
        self.data = data
        self.grad = 0.0 # default gradient is zero, represents derivative of the output w respect to value
        self._backward = lambda: None
        self._prev=set(_children)
        self._op=_op
        self.label = label

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
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out

def lol():

    h = 0.0001
    a = Value(data=2.0, label='a')
    b = Value(data=-3.0, label='b')
    c = Value(data=10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label = 'f')
    L = d*f; L.label = 'L'
    L1 = L.data

    a = Value(data=2.0, label='a')
    b = Value(data=-3.0, label='b')
    c = Value(data=10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label = 'f')
    L = d*f; L.label = 'L'
    L2 = L.data + h

    print((L2-L1)/h)

#inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

#weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias of the neuron
b = Value(6.7, label = 'b')

x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'

#sum(WiXi+b)
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'

#neuron
n = x1w1x2w2 + b; n.label='n'

#output
o = n.tanh()


if __name__ == '__main__': 
    draw_dot(o)


