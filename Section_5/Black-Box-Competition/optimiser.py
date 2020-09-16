import numpy as np

N_EVALUATIONS = 32
SEED = 122222

class Optimiser(object):
  def __init__(self, x0, scale=1e-2, seed=1122):
    self.x0 = x0
    self.rng = np.random.RandomState(seed=seed)

    self.x = x0
    self.f = None

    self.scale = scale

  def ask(self, ):
    return self.x + self.rng.normal(size=self.x.shape, scale=self.scale)

  def tell(self, x, f):
    if self.f is None:
      self.f = f
      self.x = x

    elif self.f < f:
      pass

    else:
      self.f = f
      self.x = x

  def reset(self, ):
    self.x = self.x0
    self.f = None

if __name__ == '__main__':
  from tqdm import tqdm

  from fel import SASE, random_beam, random_geometry
  from fel import eval_optimization

  rng = np.random.RandomState(seed=SEED)

  run_results = list()

  for _ in range(N_EVALUATIONS):
    sase = SASE(random_beam(rng), random_geometry(rng))
    epsilon = 1e-12

    objective = lambda x: np.log(1e-3) - np.log(sase.rho_int(x) + epsilon)

    bounds = np.stack([
        -2 * np.ones(sase.ndim()),
        +2 * np.ones(sase.ndim())
    ], axis=1)

    x0 = np.zeros(sase.ndim())

    xs, fs = eval_optimization(
      Optimiser,
      bounds=bounds,
      f=objective,
      x0=x0,
      moving_cost=1,
      measuring_cost=1,
      budget=128,
      progress=tqdm
    )

    print('Score:', np.min(fs))
    run_results.append(np.min(fs))

  print()
  print('Avg. score:', np.mean(run_results))
  print('std:', np.std(run_results))

