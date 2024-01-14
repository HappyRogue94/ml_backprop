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

        def _backward():
            self.grad  = (1.0) * out.grad
            other.grad = (1.0) * out.grad
        
        out._backward = _backward
            
        return out

    def __mul__(self, other):
        """
        this method is used to overload the '*' operator to allow for multiplication
        of two seperate objects
        """
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  = other.data * out.grad
            other.grad = self.data * out.grad 
        
        out._backward = _backward

        return out
    
    def backward(self):
        """
        automated backward propagation algorithm which returns the gradient (chain rule) at each node with
        respect to the output
        """
        # store list of nodes from topo sort algorithm
        topoList = []
        # keep track of set of nodes visited
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topoList.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topoList):
            node._backward()

    
    def tanh(self):
        """
        activation function which is used to squash the output between (-1, 1)
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        
        out._backward = _backward

        return out

if __name__ == '__main__': 
    #inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    #weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the neuron
    b = Value(6.88137358, label = 'b')

    x1w1 = x1*w1; x1w1.label='x1*w1'
    x2w2 = x2*w2; x2w2.label='x2*w2'

    #sum(WiXi+b)
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'

    #neuron
    n = x1w1x2w2 + b; n.label='n'

    #output
    o = n.tanh()



