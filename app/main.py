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

        # Define UI properties
        self.circle_radius = 50
        self.circle_center = None  # Will be calculated during paint
        self.is_frozen = False
        self.frozen_frame = None

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Mouse Event Listener')
        self.show()

    def mousePressEvent(self, event):
        if self.is_inside_circle(event.x(), event.y()):
            self.is_frozen = not self.is_frozen
            if self.is_frozen:
                self.frozen_frame = self.frame.copy()
            self.update()

    def mouseReleaseEvent(self, event):
        pass

    def is_inside_circle(self, x, y):
        distance = math.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2)
        return distance <= self.circle_radius

    def display_feed(self):
        if not self.is_frozen:
            ret, self.frame = self.capture.read()
            self.update()

    def paintEvent(self, event):
        display_frame = self.frozen_frame if self.is_frozen else self.frame
        if display_frame is not None:
            image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
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
            
            qp = QPainter(self)
            qp.drawPixmap(QRect(x_offset, y_offset, scaled_width, scaled_height), 
                         QPixmap.fromImage(qimage).scaled(scaled_width, scaled_height,
                                                        Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation))

            # Calculate circle position at bottom center of camera image
            self.circle_center = (x_offset + scaled_width // 2,
                                y_offset + scaled_height - self.circle_radius - 10)  # 10px padding from bottom
            
            # Draw either circle or X based on frozen state
            qp.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
            if not self.is_frozen:
                # Draw circle
                qp.setBrush(QColor(255, 255, 255, 128))  # Semi-transparent white
                qp.drawEllipse(self.circle_center[0] - self.circle_radius,
                              self.circle_center[1] - self.circle_radius,
                              self.circle_radius * 2, self.circle_radius * 2)
            else:
                # Draw X
                qp.setBrush(Qt.NoBrush)
                x, y = self.circle_center
                r = self.circle_radius
                qp.drawLine(x - r, y - r, x + r, y + r)
                qp.drawLine(x - r, y + r, x + r, y - r)

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
