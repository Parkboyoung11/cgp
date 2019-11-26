from collections import namedtuple
import numpy as np
from copy import deepcopy

InputNode    = namedtuple('InputNode',    ('name', 'output', 'n'))
ProgramNode  = namedtuple('ProgramNode',  ('name', 'output', 'active', 'n', 'x', 'y', 'f', 'p'))
OutputNode   = namedtuple('OutputNode',   ('name', 'output', 'x'))
FunctionNode = namedtuple('FunctionNode', ('name', 'function'))

class Individual(object):

  def __init__(self, input_nodes, output_nodes, function_nodes, C, r):
    self.input_nodes      = input_nodes
    self.program_nodes    = [ProgramNode('node_%d' % c, None, None, None, None, None, None, None) for c in range(C)]
    self.output_nodes     = output_nodes
    self.function_nodes   = function_nodes
    self.N                = len(input_nodes) + C
    self.C                = C 
    self.r                = r
    self._initialize()

  def _initialize(self):
    for i, node in enumerate(self.input_nodes):
      self.input_nodes[i]   = node._replace(n=i)
    for i, node in enumerate(self.program_nodes):
      self.program_nodes[i] = node._replace(active=False,
                                            n=len(self.input_nodes)+i, 
                                            x=np.random.rand(), 
                                            y=np.random.rand(), 
                                            f=np.random.rand(), 
                                            p=np.random.rand())
    for i, node in enumerate(self.output_nodes):
      self.output_nodes[i]  = node._replace(x=np.random.rand()) 
    self._set_active()

  @property
  def phenotype(self):
    return self.input_nodes + self.program_nodes

  def _decode_x(self, x, n):
    N, r = self.N, self.r
    return self.phenotype[int(np.floor(N * x * ((1 - n / float(N)) * r + n / N)))]

  def _decode_f(self, f):
    return self.function_nodes[int(np.floor(len(self.function_nodes) * f))].function

  def _decode_p(self, p):
    return p * 2 - 1

  def _decode(self, node):
    x        = self._decode_x(node.x, node.n)
    y        = self._decode_x(node.y, node.n)
    function = self._decode_f(node.f)
    p        = self._decode_p(node.p)
    return x, y, function, p

  def _set_active_recursive(self, node):
    if type(node) == ProgramNode and not node.active: # prevent re-assign
      # assign this node active
      i = node.n - len(self.input_nodes)
      self.program_nodes[i] = node._replace(active=True, output=0.0)
      # recursively assign child node active
      node_x = self._decode_x(node.x, node.n)
      node_y = self._decode_x(node.y, node.n)
      self._set_active_recursive(node_x)
      self._set_active_recursive(node_y)

  def _set_active(self): # call whenever change the genotype
    for i, node in enumerate(self.program_nodes):
      self.program_nodes[i] = node._replace(active=False)
    for node in self.output_nodes:
      node_x = self._decode_x(node.x, self.N)
      self._set_active_recursive(node_x)

  def _set_inputs(self, inputs):
    for i, node in enumerate(self.input_nodes):
      self.input_nodes[i] = node._replace(output=inputs[i])

  def _compute_output_program_node(self, node):
    x, y, function, p = self._decode(node)
    output            = p * function(x.output, y.output)
    return output

  def _compute_output_output_node(self, node):
    node_x = self._decode_x(node.x, self.N)
    return node_x.output

  def forward(self, inputs):
    self._set_inputs(inputs)
    for i, node in enumerate(self.program_nodes):
      if node.active:
        output                = self._compute_output_program_node(node)
        self.program_nodes[i] = node._replace(output=output)

    outputs = np.zeros(len(self.output_nodes))
    for i, node in enumerate(self.output_nodes):
      output               = self._compute_output_output_node(node)
      outputs[i]           = output
      self.output_nodes[i] = node._replace(output=output)
    return outputs

  def _active_program_node_mutate(self, pnmr):
    for i, node in enumerate(self.program_nodes):
      # if node.active:
      if np.random.rand() < pnmr:
        # attrib = np.random.choice(['x', 'y', 'f', 'p'], p=[0.1, 0.1, 0.05, 0.75])
        for attrib in ['x', 'y', 'f', 'p']:
          self.program_nodes[i] = self.program_nodes[i]._replace(**{attrib: np.random.rand()})
        self.program_nodes[i] = self.program_nodes[i]._replace(p=1)


  def _output_node_mutate(self, onmr):
    for i, node in enumerate(self.output_nodes):
      if np.random.rand() < onmr:
        x = np.clip(np.random.rand(), len(self.input_nodes) / self.N, 1)
        self.output_nodes[i] = node._replace(x=x)

  def _mutate(self, pnmr, onmr):
    self._active_program_node_mutate(pnmr)
    self._output_node_mutate(onmr)
    self._set_active()

  def mutate(self, pnmr, onmr): 
    child = deepcopy(self) # create a clone
    child._mutate(pnmr, onmr)
    return child

  @property
  def expr(self):
    for node in self.output_nodes:
      node_x = self._decode_x(node.x, self.N)
      print(node_x)
    res = []
    for node in self.program_nodes:
      if node.active:
        x, y, f, p = self._decode(node)
        res.append('%d(%d,%d,%s,%0.2f)' % (node.n, x.n, y.n, f.__name__, p))
    return ' '.join(res)

def ADD(x, y):
  return (x + y) / 2

def MUL(x, y):
  return x * y

def main():
  input_nodes    = [InputNode(name, None, None)  for name in ['x', 'dx', 'a', 'da']]
  output_nodes   = [OutputNode(name, None, None) for name in ['left', 'right']]
  function_nodes = [FunctionNode(name, function) for name, function in zip(['add', 'mul'], [ADD, MUL])]
  C    = 100
  r    = 0.1
  pnmr = 0.1
  onmr = 0.5

  inputs = np.random.rand(4)
  ind = Individual(input_nodes, output_nodes, function_nodes, C, r)
  for i in range(3):
    outputs = ind.forward(inputs)
    print(outputs)

  print(ind.expr)
  ind.mutate(pnmr, onmr)


if __name__ == '__main__':
  main()
