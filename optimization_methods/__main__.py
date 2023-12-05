import sys
from PyQt5.QtWidgets import QApplication
from optimization_methods.gui import MainWindow


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as error:
        print(f"Ошибка: {error}")
