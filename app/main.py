import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import math
import numpy as np
from ultralytics import YOLO, SAM

yolo = YOLO("yolo11n.pt")
sam = SAM("sam2_s.pt")

SEG_POINT_RADIUS = 10

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
        self.detections = None
        self.masks = []  # List to store multiple masks
        self.click_coords = []  # List of (x,y,label) tuples

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
        if self.frame is not None:
            self.detections = yolo(self.frame)[0].boxes.data

    def is_button_press(self, x, y):
        assert self.button_center is not None
        distance = math.sqrt((x - self.button_center[0])**2 + (y - self.button_center[1])**2)
        return distance <= self.button_radius

    def mousePressEvent(self, event):
        if self.is_button_press(event.x(), event.y()):
            if not self.is_frozen: self.capture()
            self.is_frozen = not self.is_frozen
            self.update()
        elif self.is_frozen and self.frame is not None:
            self.segment(event)

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

            # Calculate scale factors
            scale_x = scaled_width / width
            scale_y = scaled_height / height

            # Draw segmentation mask if available
            if self.is_frozen and self.masks:
                # Combine all masks
                combined_mask = np.zeros_like(self.masks[0])
                for mask in self.masks:
                    combined_mask = np.logical_or(combined_mask, mask)
                mask_image = combined_mask.astype(np.uint8) * 255
                mask_image = cv2.resize(mask_image, (width, height))
                mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
                mask_colored[mask_image > 0] = [0, 255, 0, 128]  # Semi-transparent green

                mask_qimage = QImage(mask_colored.data, width, height, 4 * width, QImage.Format_RGBA8888)
                qp.drawPixmap(
                    QRect(x_offset, y_offset, scaled_width, scaled_height),
                    QPixmap.fromImage(mask_qimage).scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

                # Draw all click points
                for x, y, _ in self.click_coords:
                    click_x = int(x * scale_x) + x_offset
                    click_y = int(y * scale_y) + y_offset
                    qp.setPen(QPen(QColor(255, 0, 0), 4))
                    qp.drawPoint(click_x, click_y)

            # Draw detection boxes and labels if frozen
            if self.is_frozen and self.detections is not None:
                # Scale factor for coordinates
                scale_x = scaled_width / width
                scale_y = scaled_height / height

                for detection in self.detections:
                    x1, y1, x2, y2, conf, cls = detection

                    # Scale coordinates
                    x1 = int(x1.item() * scale_x) + x_offset
                    y1 = int(y1.item() * scale_y) + y_offset
                    x2 = int(x2.item() * scale_x) + x_offset
                    y2 = int(y2.item() * scale_y) + y_offset

                    # Draw box
                    qp.setPen(QPen(QColor(255, 0, 0), 2))
                    qp.drawRect(x1, y1, x2 - x1, y2 - y1)

                    # Draw label
                    class_name = yolo.names[int(cls.item())]
                    confidence = f"{conf.item():.2f}"
                    label = f"{class_name} {confidence}"

                    qp.setPen(QPen(QColor(255, 0, 0)))
                    qp.drawText(x1, y1 - 5, label)

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

    def segment(self, event):
        # Convert click coordinates back to image space
        height, width = self.frame.shape[:2]
        window_width = self.width()
        window_height = self.height()
        image_aspect = width / height
        window_aspect = window_width / window_height

        if window_aspect > image_aspect:
            scaled_width = int(window_height * image_aspect)
            scaled_height = window_height
            x_offset = (window_width - scaled_width) // 2
            y_offset = 0
        else:
            scaled_width = window_width
            scaled_height = int(window_width / image_aspect)
            x_offset = 0
            y_offset = (window_height - scaled_height) // 2

        # Convert click to image coordinates
        img_x = (event.x() - x_offset) * (width / scaled_width)
        img_y = (event.y() - y_offset) * (height / scaled_height)

        if 0 <= img_x < width and 0 <= img_y < height:
            remove_idx = None
            for idx, (x, y, _) in enumerate(self.click_coords):
                if abs(x - img_x) <= SEG_POINT_RADIUS and abs(y - img_y) <= SEG_POINT_RADIUS:
                    remove_idx = idx
                    break

            if remove_idx is not None:
                # Remove the point
                self.click_coords.pop(remove_idx)
            else:
                # Add new point
                self.click_coords.append((img_x, img_y, 1))

            # Run SAM prediction with all points
            if self.click_coords:
                points = [[x, y] for x, y, _ in self.click_coords]
                # labels = [label for _, _, label in self.click_coords]
                results = sam(self.frame, points=points)#, labels=labels)
                print(len(results))
                print(len(results[0].masks))
                self.masks = [mask.cpu().numpy() for mask in results[0].masks.data]
            else:
                self.masks = []

            self.update()

    def closeEvent(self, event):
        self.camera.release()
        self.timer.stop()

def main():
    app = QApplication(sys.argv)
    listener = MouseEventListener()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
