import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from typing import Callable


class OptimizationMethod:
    def __init__(self, eps: float, max_iterations: int):
        self.eps = eps
        self.max_iterations = max_iterations
        self.step_logs = ""

    def _add_step_message(self, message: str):
        self.step_logs += message

    @staticmethod
    def _calculate_gradient(f: Callable, x: np.array, eps: float = 0.00001) -> np.array:
        def df_dxi(i: int) -> float:
            d = np.zeros(x.shape)
            d[i] = eps
            return (f(x + d) - f(x - d)) / (2 * eps)

        return np.array([df_dxi(i) for i in range(len(x))])

    @staticmethod
    def _calculate_inv_hessian(f: Callable, x: np.array, eps: float = 0.00001) -> np.array:
        def df_dxi(x: np.array, i: int) -> float:
            d = np.zeros(x.shape)
            d[i] = eps
            return (f(x + d) - f(x - d)) / (2 * eps)

        def df_dxixj(x: np.array, i: int = 0, j: int = 0) -> float:
            d = np.zeros(x.shape)
            d[j] = eps
            return (df_dxi(x + d, i) - df_dxi(x - d, i)) / (2 * eps)

        n = len(x)
        matrix_arrays = np.array([[df_dxixj(x, i, j) for j in range(n)] for i in range(n)])
        return np.linalg.inv(np.array(matrix_arrays))

    def optimize(self, f: Callable, x0: np.array) -> tuple:
        pass


class GradientDescentMethod(OptimizationMethod):
    def __init__(self, eps: float, max_iterations: int):
        super().__init__(eps, max_iterations)

    @staticmethod
    def _calculate_ak(f: Callable, xk: np.array, df_dx_k: np.array) -> float:
        alpha0 = np.array([0])
        return minimize(lambda alpha: f(xk - alpha * df_dx_k), alpha0, method='nelder-mead').x

    def optimize(self, f: Callable, x0: np.array) -> (np.array, int):
        self._add_step_message(f"Начало работы метода. Начальная точка x0 = {x0}.\n\n")

        k = 0
        x = [x0]
        a = []
        df_dx = []

        while k < self.max_iterations:
            df_dx.append(self._calculate_gradient(f, x[k]))
            if norm(df_dx[k]) < self.eps:
                break

            a.append(self._calculate_ak(f, x[k], df_dx[k]))
            x.append(x[k] - a[k] * df_dx[k])

            if np.sum(x[k]) != 0 and norm(x[k + 1] - x[k]) / norm(x[k]) < self.eps:
                break

            self._add_step_message(f"[Шаг k={k + 1}]\n"
                                   f"    f(x_k) = {f(x[k + 1])}\n"
                                   f"    f(x_k) call count = {f.call_count - (k + 1)}\n"
                                   f"    x_k = {x[k + 1]}\n"
                                   f"    a_k = {a[k]}\n"
                                   f"    grad(x_k) = {df_dx[k]}\n\n")

            k += 1

        self._add_step_message(f"Завершение работы метода. ")
        if k >= self.max_iterations:
            self._add_step_message(f"Превышено максимальное количество итераций. Последняя найденная точка x_k = {x[k]}. Значение f(x_k) = {f(x[k])}.\n\n")
        else:
            self._add_step_message(f"Оптимальная точка найдена. x* = {x[k]}. Значение f(x*) = {f(x[k])}.\n\n")

        return x[k], k


class ConjugateGradientsMethod(OptimizationMethod):
    def __init__(self, eps: float, max_iterations: int):
        super().__init__(eps, max_iterations)

    @staticmethod
    def _calculate_ak(f: Callable, xk: np.array, sk: np.array):
        alpha0 = np.array([0])
        return minimize(lambda alpha: f(xk - alpha * sk), alpha0, method='nelder-mead').x

    def optimize(self, f: Callable, x0: np.array) -> (np.array, int):
        self._add_step_message(f"Начало работы метода. Начальная точка x0 = {x0}.\n\n")

        k = 0
        n = len(x0)
        x = [x0]
        a = []
        s = []
        df_dx = []

        while k < self.max_iterations:
            df_dx.append(self._calculate_gradient(f, x[k]))
            if norm(df_dx[k]) < self.eps:
                break

            if k == 0:
                s.append(-df_dx[k])
            else:
                s.append(-df_dx[k] + s[k - 1] * (np.dot(df_dx[k], df_dx[k]) / np.dot(df_dx[k - 1], df_dx[k - 1])))

            a.append(self._calculate_ak(f, x[k], s[k]))
            x.append(x[k] - a[k] * s[k])

            self._add_step_message(f"[Шаг k={k+1}]\n"
                                   f"    x_k = {x[k+1]}\n"
                                   f"    f(x_k) = {f(x[k+1])}\n"
                                   f"    f(x_k) call count = {f.call_count - (k+1)}\n"
                                   f"    a_k = {a[k]}\n"
                                   f"    s_k = {s[k]}\n"
                                   f"    grad(x_k) = {df_dx[k]}\n\n")

            k += 1

            if norm(x[k] - x[k - 1]) < self.eps or k == n:
                break

        self._add_step_message(f"Завершение работы метода. ")
        if k >= self.max_iterations:
            self._add_step_message(
                f"Превышено максимальное количество итераций. Последняя найденная точка x_k = {x[k]}. Значение f(x_k) = {f(x[k])}.\n\n")
        else:
            self._add_step_message(f"Оптимальная точка найдена. x* = {x[k]}. Значение f(x*) = {f(x[k])}.\n\n")

        return x[k], k


class NewtonMethod(OptimizationMethod):
    def __init__(self, eps: float, max_iterations: int):
        super().__init__(eps, max_iterations)

    def optimize(self, f: Callable, x0: np.array) -> (np.array, int):
        self._add_step_message(f"Начало работы метода. Начальная точка x0 = {x0}.\n\n")

        k = 0
        x = [x0]
        h = []
        s = []
        df_dx = []

        while k < self.max_iterations:
            df_dx.append(self._calculate_gradient(f, x[k]))
            if norm(df_dx[k]) < self.eps:
                break

            h.append(self._calculate_inv_hessian(f, x[k]))
            s.append(np.array(np.transpose(np.negative(np.matmul(h[k], df_dx[k])))).ravel())
            x.append(x[k] + s[k])

            if np.sum(x[k]) != 0 and norm(x[k + 1] - x[k]) / norm(x[k]) < self.eps:
                break

            t = str(h[k]).replace("\n", "\n                   ")
            self._add_step_message(f"[Шаг k={k + 1}]\n"
                                   f"    x_k = {x[k + 1]}\n"
                                   f"    f(x_k) = {f(x[k + 1])}\n"
                                   f"    f(x_k) call count = {f.call_count - (k + 1)}\n"
                                   f"    s_k = {s[k]}\n"
                                   f"    H(x_k) = {t}\n"
                                   f"    grad(x_k) = {df_dx[k]}\n\n")
            k += 1

        self._add_step_message(f"Завершение работы метода. ")
        if k >= self.max_iterations:
            self._add_step_message(
                f"Превышено максимальное количество итераций. Последняя найденная точка x_k = {x[k]}. Значение f(x_k) = {f(x[k])}.\n\n")
        else:
            self._add_step_message(f"Оптимальная точка найдена. x* = {x[k]}. Значение f(x*) = {f(x[k])}.\n\n")

        return x[k], k
