import re
import sympy as sp
import numpy as np


class Function:
    def __init__(self, function_str: str):
        self.arg_count = self._calculate_arg_count(function_str)
        self.function = sp.sympify(function_str)
        self.call_count = 0

    @staticmethod
    def _calculate_arg_count(function_str: str) -> int:
        pattern = re.compile("x\\d+")
        results = re.findall(pattern, function_str)
        indexes = set([int(r[1:]) for r in results])
        return max(indexes)

    def __call__(self, x: np.array) -> float:
        self.call_count += 1
        args = {}
        for i in range(0, len(x)):
            args[f"x{i+1}"] = x[i]
        return float(self.function.subs(args))
