import numpy as np
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from optimization_methods.common import Function
from optimization_methods.common import GradientDescentMethod, NewtonMethod, ConjugateGradientsMethod


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        ui_filepath = str(Path(__file__).parent.absolute().joinpath(f"{self.__class__.__name__}.ui"))
        uic.loadUi(ui_filepath, self)

        self.startButton.pressed.connect(self._start_method_execution)

    def _set_results(self, results: str):
        self.resultsTextBrowser.setText(results)

    def _set_step_logs(self, logs: str):
        self.stepsTextBrowser.setText(logs)

    def _start_method_execution(self):
        self._set_results("")
        self._set_step_logs("")

        function_str = self.functionEdit.text()
        try:
            function = Function(function_str)
        except:
            self._set_results(f"Ошибка. Не удалось распознать целевую функцию.\n"
                              f"Проверьте корректность ввода целевой функции.\n")
            return

        x0_str = self.xEdit.text()
        try:
            x0 = np.array(np.fromstring(x0_str, dtype=float, sep=","))
        except:
            self._set_results(f"Ошибка. Не удалось распознать начальную точку.\n"
                              f"Проверьте корректность ввода начальной точки.\n")
            return

        if function.arg_count != len(x0):
            self._set_results(f"Ошибка. Количество аргументов целевой функции f не совпадает с размерностью "
                              f"начальной точки x0.\nПроверьте корректность ввода исходных данных.")
            return

        max_iterations = int(self.iterationSpinBox.value())
        eps = float(self.epsilonSpinBox.value())

        if self.methodComboBox.currentIndex() == 0:
            method_class = GradientDescentMethod
        elif self.methodComboBox.currentIndex() == 1:
            method_class = NewtonMethod
        elif self.methodComboBox.currentIndex() == 2:
            method_class = ConjugateGradientsMethod
        else:
            self._set_results(f"Выбран неизвестный метод.\n")
            return

        method = method_class(eps, max_iterations)

        try:
            x, steps = method.optimize(function, x0)
        except Exception as error:
            self._set_results(f"При работе метода возникла ошибка: {error}\nПроверьте корректность ввода исходных данных.\n")
            return

        self._set_step_logs(method.step_logs)

        if steps >= method.max_iterations:
            self._set_results(f"Достигнуто максимальное количество итераций. Последняя точка x_k = {x}. Значение f(x_k) = {function(x)}.\n")
        else:
            self._set_results(f"Оптимальное значение найдено. x* = {x}\n"
                              f"Значение f(x*) = {function(x)}\n"
                              f"Количество итераций: {steps}\n")
