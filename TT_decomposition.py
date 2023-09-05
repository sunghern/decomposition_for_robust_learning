import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from joblib import Parallel, delayed
from multiprocessing import Process, Manager, cpu_count, Pool


class SolveIndividual:
    def solve(self, A, b, nu, rho, Z):
        t1 = A.dot(A.T)
        A = A.reshape(-1, 1)
        tX = (A * b + rho * Z - nu) / (t1 + rho)
        return tX