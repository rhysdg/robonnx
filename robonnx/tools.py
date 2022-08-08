
import re
import time
import pickle 

class Timer:
  def __init__(self, verbose=False):
    self.verbose = verbose
    self.start()

  def start(self, operation='default process'):
    self.st = time.monotonic()
    self.operation = operation

  def end(self):
    end = time.monotonic()
    cost = f'{(end-self.st)*1000:.2f}ms'
    pickle.dump(cost,open(f'{re.sub(" ", "_", self.operation)}.pkl', 'wb'))
    if self.verbose:
      print(f'{self.operation} in:  {cost}')