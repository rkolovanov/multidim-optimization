import unittest
import numpy as np
from numpy.linalg import norm
from optimization_methods.common.methods import NewtonMethod
from optimization_methods.common.function import Function


class TestNewtonMethod(unittest.TestCase):
    def test_simple_1(self):
        f = Function("(x1 + 2) ** 2 + (x2 - 3) ** 2 + (x2 + 3) ** 2 - 3 * x1")
        nm = NewtonMethod(0.00001, 10000)
        x0 = np.array([-5, -10])
        x, k = nm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-0.5, 0])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_2(self):
        f = Function("(x1 + 2) ** 2 + (x2 - 3) ** 2 + (x2 + 3) ** 2 - 3 * x1")
        nm = NewtonMethod(0.00001, 10000)
        x0 = np.array([20, -17])
        x, k = nm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-0.5, 0])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_3(self):
        f = Function("(x1 + 2) ** 2 + (x2 - 3) ** 2 + (x2 + 3) ** 2 - 3 * x1")
        nm = NewtonMethod(0.00001, 10000)
        x0 = np.array([43, 19])
        x, k = nm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-0.5, 0])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_1(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        nm = NewtonMethod(0.01, 10000)
        x0 = np.array([-30, 24])
        x, k = nm.optimize(f, x0)
        eps = 0.02
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_2(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        nm = NewtonMethod(0.001, 10000)
        x0 = np.array([-30, 24])
        x, k = nm.optimize(f, x0)
        eps = 0.001
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_3(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        nm = NewtonMethod(0.0001, 10000)
        x0 = np.array([-30, 24])
        x, k = nm.optimize(f, x0)
        eps = 0.0001
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_already_min(self):
        f = Function('(x1 - 3) ** 2 + (x2 + 7) ** 2 - 3 * x1')
        nm = NewtonMethod(0.00001, 10000)
        x0 = np.array([4.5, -7])
        x, k = nm.optimize(f, x0)
        eps = 0.00001
        test = np.array([4.5, -7])
        self.assertTrue((norm(x - test) < eps).all())

    def test_far_min(self):
        f = Function('(x1 + 6) ** 2 + (x2 - 1) ** 2 + 5 * x1')
        nm = NewtonMethod(0.00001, 10000)
        x0 = np.array([1000, 1000])
        x, k = nm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-8.5, 1])
        self.assertTrue((norm(x - test) < eps).all())
