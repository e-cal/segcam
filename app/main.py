import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import math
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

class MouseEventListener(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.camera = cv2.VideoCapture(0)
        self.frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_feed)
        self.timer.start(50)

        # UI properties
        self.button_radius = 50
        self.button_center = None
        self.is_frozen = False
        self.frozen_frame = None

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Mouse Event Listener')
        self.show()

    def display_feed(self):
        if not self.is_frozen:
            ret, self.frame = self.camera.read()
            self.update()

    def capture(self):
        self.display_feed()
        print(self.frame)

    def is_button_press(self, x, y):
        assert self.button_center is not None
        distance = math.sqrt((x - self.button_center[0])**2 + (y - self.button_center[1])**2)
        return distance <= self.button_radius

    def mousePressEvent(self, event):
        if self.is_button_press(event.x(), event.y()):
            if not self.is_frozen: self.capture()
            self.is_frozen = not self.is_frozen
            self.update()

    def mouseReleaseEvent(self, event):
        pass

    def paintEvent(self, event):
        if self.frame is not None:
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape

            # Calculate scaling to maintain aspect ratio
            window_width = self.width()
            window_height = self.height()
            image_aspect = width / height
            window_aspect = window_width / window_height
            if window_aspect > image_aspect:
                # Window is wider than image
                scaled_width = int(window_height * image_aspect)
                scaled_height = window_height
                x_offset = (window_width - scaled_width) // 2
                y_offset = 0
            else:
                # Window is taller than image
                scaled_width = window_width
                scaled_height = int(window_width / image_aspect)
                x_offset = 0
                y_offset = (window_height - scaled_height) // 2

            # Draw camera image
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            qp = QPainter(self)
            qp.drawPixmap(
                QRect(x_offset, y_offset, scaled_width, scaled_height),
                QPixmap.fromImage(qimage).scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Calculate button position at bottom center of camera image
            self.button_center = (x_offset + scaled_width // 2, y_offset + scaled_height - self.button_radius - 10)

            qp.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
            if not self.is_frozen:
                # Draw circle
                qp.setBrush(QColor(255, 255, 255, 128))
                qp.drawEllipse(
                    self.button_center[0] - self.button_radius,
                    self.button_center[1] - self.button_radius,
                    self.button_radius * 2,
                    self.button_radius * 2,
                )
            else:
                # Draw X
                qp.setBrush(Qt.NoBrush)
                x, y = self.button_center
                r = self.button_radius
                qp.drawLine(x - r, y - r, x + r, y + r)
                qp.drawLine(x - r, y + r, x + r, y - r)

    def closeEvent(self, event):
        self.camera.release()
        self.timer.stop()

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
