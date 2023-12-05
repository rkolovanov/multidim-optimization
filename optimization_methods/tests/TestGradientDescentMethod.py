import unittest
import numpy as np
from numpy.linalg import norm
from optimization_methods.common.methods import GradientDescentMethod
from optimization_methods.common.function import Function


class TestGradientDescentMethod(unittest.TestCase):
    def test_simple_1(self):
        f = Function('(x1 - 10) ** 2 + (x2 + 5) ** 2 + x1*x2')
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([1, 1])
        x, k = gdm.optimize(f, x0)
        eps = 0.0001
        test = np.array([16.66666666, -13.33333333])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_2(self):
        f = Function('(x1 - 10) ** 2 + (x2 + 5) ** 2 + x1*x2')
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([-8, 19])
        x, k = gdm.optimize(f, x0)
        eps = 0.0001
        test = np.array([16.66666666, -13.33333333])
        self.assertTrue((norm(x - test) < eps).all())

    def test_simple_3(self):
        f = Function('(x1 - 10) ** 2 + (x2 + 5) ** 2 + x1*x2')
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([13, 21])
        x, k = gdm.optimize(f, x0)
        eps = 0.001
        test = np.array([16.66666666, -13.33333333])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_1(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        gdm = GradientDescentMethod(0.01, 10000)
        x0 = np.array([15, -10])
        x, k = gdm.optimize(f, x0)
        eps = 0.02
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_2(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        gdm = GradientDescentMethod(0.001, 10000)
        x0 = np.array([15, -10])
        x, k = gdm.optimize(f, x0)
        eps = 0.002
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_different_eps_3(self):
        f = Function("(x1 + x2) ** 2 + (x1 + 1) ** 2 + (x2 - 2) ** 2")
        gdm = GradientDescentMethod(0.0001, 10000)
        x0 = np.array([15, -10])
        x, k = gdm.optimize(f, x0)
        eps = 0.001
        test = np.array([-1.33333333, 1.66666666])
        self.assertTrue((norm(x - test) < eps).all())

    def test_already_min(self):
        f = Function('(x1) ** 2 + (x2) ** 2 + (x3) ** 2')
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([0, 0, 0])
        x, k = gdm.optimize(f, x0)
        eps = 0.00001
        test = np.array([0, 0, 0])
        self.assertTrue((norm(x - test) < eps).all())

    def test_far_min(self):
        f = Function('(x1 + 2) ** 2 + (x2) ** 2 - 2 * x1')
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([1000, 1000])
        x, k = gdm.optimize(f, x0)
        eps = 0.00001
        test = np.array([-1, 0])
        self.assertTrue((norm(x - test) < eps).all())

    def test_local_min(self):
        f = lambda x: x[0] * x[1] * np.sin(x[1])
        gdm = GradientDescentMethod(0.00001, 10000)
        x0 = np.array([0, 0])
        x, k = gdm.optimize(f, x0)
        eps = 0.00001
        test = np.array([0, 0])
        self.assertTrue((norm(x - test) < eps).all())
