import unittest
import numpy as np
from numpy.linalg import norm
from optimization_methods.common.methods import GradientDescentMethod
from optimization_methods.common.methods import NewtonMethod
from optimization_methods.common.methods import ConjugateGradientsMethod
from optimization_methods.common.function import Function


if __name__ == "__main__":
    x0 = [np.array([0, 0]), np.array([-4, 6]), np.array([2, 4])]
    a = [0.1, 10]
    for i in range(len(x0)):
        for j in range(len(a)):
            f = Function("((x2 - (x1)) ** 2) + " + str(a[j]) + " * (x1 - 1) ** 2")
            method = ConjugateGradientsMethod(0.001, 10000)
            x, k = method.optimize(f, x0[i])
            print(f"a = " + str(a[j]) + "\n"
                  f"x0 = : " + str(x0[i]) + "\n"
                  f"x* = : " + str(x) + "\n"
                  f"f(x*) = " + str(f(x)) + "\n"
                  f"call_count: " + str(f.call_count - 2 - k) + "\n"
                  f"step_count: " + str(k))
            print(method.step_logs)
