import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import math

class MouseEventListener(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.capture = cv2.VideoCapture(0)
        self.frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_feed)
        self.timer.start(50)

        # Define circle properties
        self.circle_radius = 50
        self.circle_center = (400, 550)  # Bottom middle of the 800x600 window

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Mouse Event Listener')
        self.show()

    def mousePressEvent(self, event):
        if self.is_inside_circle(event.x(), event.y()):
            print(f"Mouse Press Event at ({event.x()}, {event.y()})")

    def mouseReleaseEvent(self, event):
        if self.is_inside_circle(event.x(), event.y()):
            print(f"Mouse Release Event at ({event.x()}, {event.y()})")

    def is_inside_circle(self, x, y):
        distance = math.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2)
        return distance <= self.circle_radius

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

            # Draw white circle
            qp.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
            qp.setBrush(QColor(255, 255, 255, 128))  # Semi-transparent white
            qp.drawEllipse(self.circle_center[0] - self.circle_radius, 
                           self.circle_center[1] - self.circle_radius, 
                           self.circle_radius * 2, self.circle_radius * 2)

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
