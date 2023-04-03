import numpy as np
import matplotlib.pyplot as plt

class Swarm:
    def __init__(self, n, dim, bounds, fun) -> None:
        self.n = n
        self.particles = np.zeros((n, dim))

        for idx, b in enumerate(bounds):
            self.particles[:,idx] = np.random.uniform(b[0], b[1], n)

        print(self.particles)
        # plt.scatter(self.particles[:,0], self.particles[:,1])
        # plt.show()


if __name__ == "__main__":
    def fun(x):
        x1,x2 = x
        return 0.1*x1**2 + 0.1*x2**2 - np.cos(3*x1) - np.cos(3*x2)
    
    n = 100
    dim = 2

    bounds = [[-10,10], [-10,10]]
    optimizer = Swarm(n, dim, bounds, fun)
