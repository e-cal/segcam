from collections import namedtuple
from enum import Enum
import sys
from typing import Literal, NamedTuple
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2
import math
import numpy as np
from ultralytics import YOLO, SAM
import torch
from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
from dataclasses import dataclass

yolo = YOLO("yolo11n.pt")
# sam = SAM("sam2_s.pt")
sam2 = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"))


SEG_POINT_RADIUS = 10

Point = namedtuple("Point", ["x", "y"])

@dataclass
class WindowScaling:
    scaled_width: int
    scaled_height: int 
    x_offset: int
    y_offset: int
    scale_x: float
    scale_y: float

class MaskColor(Enum):
    GREEN = (0, 128, 0)
    BLUE = (0, 0, 255)
    PINK = (238, 84, 144)

    @classmethod
    def get_next_color(cls, color):
        colors = list(cls)
        if color not in colors:
            raise ValueError("Invalid color")
        index = colors.index(color)
        return colors[(index + 1) % len(colors)]

    @property
    def rgb(self):
        return self.value

@dataclass
class Mask:
    point: Point # click coordinates
    masks: np.ndarray # computed masks for the current point
    active: int # index of the active mask
    label: Literal[0, 1] = 1 # background (0) or foreground (1)
    color: MaskColor = MaskColor.GREEN

    @property
    def active_mask(self):
        return self.masks[self.active]


class MouseEventListener(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.camera = cv2.VideoCapture(0)
        self.frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_img)
        self.timer.start(50)

        # UI properties
        self.button_radius = 50
        self.button_center = None
        self.is_frozen = False
        self.frozen_frame = None
        self.detections = None
        self.masks: list[Mask] = []

    @property
    def seg_points(self):
        return [mask.point for mask in self.masks]

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('SegCam')
        self.show()

    def mousePressEvent(self, event):
        if self.is_button_press(event.x(), event.y()):
            if not self.is_frozen: self.freeze()
            self.is_frozen = not self.is_frozen
            self.update()
        elif self.is_frozen and self.frame is not None:
            self.segment(event)

    def is_button_press(self, x, y):
        assert self.button_center is not None
        distance = math.sqrt((x - self.button_center[0])**2 + (y - self.button_center[1])**2)
        return distance <= self.button_radius

    def update_img(self):
        if not self.is_frozen:
            ret, self.frame = self.camera.read()
            self.update()

    def freeze(self):
        self.update_img()
        if self.frame is not None:
            self.detections = yolo(self.frame)[0].boxes.data

    def _get_img_scaling(self, image):
        # Calculate scaling to maintain aspect ratio
        window_width = self.width()
        window_height = self.height()
        height, width, channel = image.shape
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

        return scaled_width, scaled_height, x_offset, y_offset, scaled_width / width, scaled_height / height

    def paintEvent(self, event):
        if self.frame is not None:
            # Get the current camera frame
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            scaled_width, scaled_height, x_offset, y_offset, scale_x, scale_y = self._get_img_scaling(image)
            self.scaling = WindowScaling(scaled_width, scaled_height, x_offset, y_offset, scale_x, scale_y)

            # Draw camera image
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            qp = QPainter(self)
            qp.drawPixmap(
                QRect(x_offset, y_offset, scaled_width, scaled_height),
                QPixmap.fromImage(qimage).scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Draw segmentation mask if available
            if self.is_frozen and self.masks:
                self.draw_masks(qp, height, width)

            # Draw detection boxes and labels if frozen
            if self.is_frozen:
                self.draw_detections(qp)

            # Calculate button position at bottom center of camera image
            self.button_center = (x_offset + scaled_width // 2, y_offset + scaled_height - self.button_radius - 10)

            # Draw freeze/unfreeze button
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


    def draw_masks(self, qp, height, width):
        if not self.masks:
            return
            
        # Combine all masks
        combined_mask = np.zeros_like(self.masks[0].active_mask)
        for mask in self.masks:
            combined_mask = np.logical_or(combined_mask, mask.active_mask)
        mask_image = combined_mask.astype(np.uint8) * 255
        mask_image = cv2.resize(mask_image, (width, height))
        mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
        mask_colored[mask_image > 0] = [0, 255, 0, 128]  # Semi-transparent green

        mask_qimage = QImage(mask_colored.data, width, height, 4 * width, QImage.Format_RGBA8888)
        qp.drawPixmap(
            QRect(self.scaling.x_offset, self.scaling.y_offset, self.scaling.scaled_width, self.scaling.scaled_height),
            QPixmap.fromImage(mask_qimage).scaled(self.scaling.scaled_width, self.scaling.scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # Draw all segmentation points
        for x, y in [mask.point for mask in self.masks]:
            _x = int(x * self.scaling.scale_x) + self.scaling.x_offset
            _y = int(y * self.scaling.scale_y) + self.scaling.y_offset
            qp.setPen(QPen(QColor(255, 0, 0), 4))
            qp.drawPoint(_x, _y)

    def draw_detections(self, qp):
        if not self.detections is not None:
            return
            
        qp.setPen(QPen(QColor(255, 0, 0), 2))
        for detection in self.detections:
            x1, y1, x2, y2, conf, cls = detection

            # Scale coordinates
            x1 = int(x1.item() * self.scaling.scale_x) + self.scaling.x_offset
            y1 = int(y1.item() * self.scaling.scale_y) + self.scaling.y_offset
            x2 = int(x2.item() * self.scaling.scale_x) + self.scaling.x_offset
            y2 = int(y2.item() * self.scaling.scale_y) + self.scaling.y_offset

            # Draw box
            qp.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Draw label
            class_name = yolo.names[int(cls.item())]
            confidence = f"{conf.item():.2f}"
            label = f"{class_name} {confidence}"
            qp.drawText(x1, y1 - 5, label)

    def _convert_for_segmentation(self, event):
        # Convert click coordinates back to image space
        assert self.frame is not None
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
        point = Point(img_x, img_y)

        return point, width, height

    def segment(self, event):
        point, width, height = self._convert_for_segmentation(event)
        if not ((0 <= point.x < width) and (0 <= point.y < height)): return

        # check if we clicked on a pre-existing point
        repeat_idx = None
        for idx, (x, y) in enumerate(self.seg_points):
            if abs(x - point.x) <= SEG_POINT_RADIUS and abs(y - point.y) <= SEG_POINT_RADIUS:
                repeat_idx = idx
                break

        if repeat_idx is not None: # cycle mask
            mask = self.masks[repeat_idx]
            if mask.active < len(mask.masks) - 1:
                pass
            else:
                self.masks.pop(repeat_idx)
        else: # new mask
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam2.set_image(self.frame)
                masks, _, _ = sam2.predict([point], [1])
            print(f"{len(masks)} masks for {point}")
            self.masks.append(Mask(point, masks, 0))

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
