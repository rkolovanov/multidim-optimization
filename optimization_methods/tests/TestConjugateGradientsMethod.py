import unittest
import numpy as np
from numpy.linalg import norm
from optimization_methods.common.methods import ConjugateGradientsMethod
from optimization_methods.common.function import Function


class TestConjugateGradientsMethod(unittest.TestCase):
    def test_simple_1(self):
        f = Function('(x1 + 2) ** 2 + (x1 - 3) ** 2 + (x2 + 3) ** 2')
        cgm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([4, -3])
        x, k = cgm.optimize(f, x0)
        eps = 0.0002
        test = np.array([0.5, -3])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_2(self):
        f = Function("(x1 + 2) ** 2 + (x1 - 3) ** 2 + (x2 + 3) ** 2")
        cgm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([-9, -5])
        x, k = cgm.optimize(f, x0)
        eps = 0.002
        test = np.array([0.5, -3])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_3(self):
        f = Function("(x1 + 2) ** 2 + (x1 - 3) ** 2 + (x2 + 3) ** 2")
        cgm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([16, 10])
        x, k = cgm.optimize(f, x0)
        eps = 0.003
        test = np.array([0.5, -3])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_1(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        cgm = ConjugateGradientsMethod(0.01, 10000)
        x0 = np.array([25, 25])
        x, k = cgm.optimize(f, x0)
        eps = 0.01
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_2(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        cgm = ConjugateGradientsMethod(0.001, 10000)
        x0 = np.array([25, 25])
        x, k = cgm.optimize(f, x0)
        eps = 0.003
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_3(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        cgm = ConjugateGradientsMethod(0.0001, 10000)
        x0 = np.array([25, 25])
        x, k = cgm.optimize(f, x0)
        eps = 0.003
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_hessian_positive(self):
        f = Function('(x1 + 6) ** 2 + (x2 - 1) ** 2 + 5 * x1')
        cgm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([100000, -100000])
        x, k = cgm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-8.5, 1])
        self.assertTrue((norm(x - test) < eps).all() and k <= len(x0))

    def test_already_min(self):
        f = Function('(x1 - 3) ** 2 + (x2 + 7) ** 2 - 3* x1')
        gdm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([4.5, -7])
        x, k = gdm.optimize(f, x0)
        eps = 0.00001
        test = np.array([4.5, -7])
        self.assertTrue((norm(x - test) < eps).all())

    def test_far_min(self):
        f = Function('(x1 + x2) ** 2 + (x2 - 6) ** 2 + 9 * x1')
        cgm = ConjugateGradientsMethod(0.00001, 10000)
        x0 = np.array([1000, -1000])
        x, k = cgm.optimize(f, x0)
        eps = 0.001
        test = np.array([-14.995, 10.5])
        self.assertTrue((norm(x - test) < eps).all())
