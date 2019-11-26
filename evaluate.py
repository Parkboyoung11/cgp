import numpy as np

data = {}
tasks = ['F%d' % _ for _ in range(1, 6)]
for task in tasks:
  data[task] = np.load('data/%s.npz' % task)

def rmse(ind, task):
  x, y   = data[task]['x'], data[task]['y']
  y_pred = np.array([ind.forward([_]) for _ in x])
  return np.sqrt(np.mean(np.power(y.ravel() - y_pred.ravel(), 2)))

def l1(ind, task):
  x, y   = data[task]['x'], data[task]['y']
  y_pred = np.array([ind.forward([_]) for _ in x])
  return np.sum(np.abs(y - y_pred))

def main():
  from individual import Individual, InputNode, OutputNode, FunctionNode
  input_nodes    = [InputNode(name, None, None)  for name in ['x']]
  output_nodes   = [OutputNode(name, None, None) for name in ['y']]
  function_nodes = [FunctionNode(name, function) for name, function in zip(['add', 'mul'], [np.add, np.multiply])]
  C = 10000
  r = 0.1

  ind     = Individual(input_nodes, output_nodes, function_nodes, C, r)
  fitness = evaluate(ind, 'F1')

  print(fitness)

if __name__ == '__main__':
  main()
