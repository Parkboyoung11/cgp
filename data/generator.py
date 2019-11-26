import numpy as np

def F1(x):
  return np.power(x, 3) + np.power(x, 2) + x

def F2(x):
  return np.power(x, 4) + F1(x)

def F3(x):
  return np.power(x, 5) + F2(x)

def F4(x):
  return np.power(x, 5) - 2 * np.power(x, 3) + x

def F5(x):
  return np.sin(np.power(x, 2)) * np.cos(x) - 1

def main():
  functions  = [F1, F2, F3, F4, F5]
  num_sample = 200

  for function in functions:
    x = np.random.rand(num_sample) * 2 - 1
    y = function(x)
    np.savez('%s.npz' % function.__name__, **{'x': x, 'y': y})

if __name__ == '__main__':
  main()