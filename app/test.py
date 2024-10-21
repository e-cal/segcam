import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter
import cv2

class MouseEventListener(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.capture = cv2.VideoCapture(0)
        self.frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_feed)
        self.timer.start(50)

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Mouse Event Listener')
        self.show()

    def mousePressEvent(self, event):
        print(f"Mouse Press Event at ({event.x()}, {event.y()})")

    def mouseReleaseEvent(self, event):
        print(f"Mouse Release Event at ({event.x()}, {event.y()})")

    def display_feed(self):
        ret, self.frame = self.capture.read()
        self.update()

    def paintEvent(self, event):
        if self.frame is not None:
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            qp = QPainter(self)
            qp.drawPixmap(QRect(0, 0, 800, 600), QPixmap.fromImage(qimage).scaled(800, 600))

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
