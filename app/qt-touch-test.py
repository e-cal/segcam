import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt

class MouseEventListener(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Mouse Event Listener')
        self.show()

    def mousePressEvent(self, event):
        print(f"Mouse Press Event at ({event.x()}, {event.y()})")

    def mouseReleaseEvent(self, event):
        print(f"Mouse Release Event at ({event.x()}, {event.y()})")

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
