import numpy as np

from evaluate import rmse, l1
from individual import Individual, InputNode, OutputNode, FunctionNode, ADD, MUL

def ea(task, best, lambda_, pnmr, onmr, T):
  print('initial', rmse(best, task))
  for t in range(T):
    offspring  = [best.mutate(pnmr, onmr) for _ in range(lambda_)]
    population = offspring + [best]
    fitness    = [rmse(ind, task) for ind in population]
    best       = population[np.argmin(fitness)]
    print(t, lambda_ * (t + 1), np.min(fitness))
    print(best.expr)

def main():
  input_nodes    = [InputNode(name, None, None)  for name in ['x']]
  output_nodes   = [OutputNode(name, None, None) for name in ['y']]
  function_nodes = [FunctionNode(name, function) for name, function in zip(['add', 'mul'], [ADD, MUL])]
  C = 50
  r = 0.0

  pnmr, onmr = 0.1, 0.2
  lambda_ = 50
  T = 200
  ind = Individual(input_nodes, output_nodes, function_nodes, C, r)
  ea('F1', ind, lambda_, pnmr, onmr, T)

if __name__ == '__main__':
  main()
