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
    GREEN = (0, 255, 115)
    BLUE = (33, 150, 243)
    PINK = (233, 30, 99)

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
    point: Point  # click coordinates
    masks: np.ndarray  # computed masks for the current point
    active: int  # index of the active mask
    label: Literal[0, 1] = 1  # background (0) or foreground (1)
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
        self.button_radius = 40
        self.freeze_button_center = None
        self.clear_button_rect = None
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
        if self.is_freeze_button_press(event.x(), event.y()):
            if not self.is_frozen: self.freeze()
            self.is_frozen = not self.is_frozen
            self.update()
        elif self.is_clear_button_press(event.x(), event.y()):
            self.masks.clear()
            self.update()
        elif self.is_frozen and self.frame is not None:
            if event.button() == Qt.RightButton:
                self.toggle_point_label(event)
            else:
                self.segment(event)

    def is_freeze_button_press(self, x, y):
        assert self.freeze_button_center is not None
        distance = math.sqrt((x - self.freeze_button_center[0])**2 + (y - self.freeze_button_center[1])**2)
        return distance <= self.button_radius

    def is_clear_button_press(self, x, y):
        if not self.is_frozen or self.clear_button_rect is None:
            return False
        return self.clear_button_rect.contains(x, y)

    def update_img(self):
        if not self.is_frozen:
            ret, self.frame = self.camera.read()
            self.update()

    def freeze(self):
        self.update_img()
        if self.frame is not None:
            self.detections = yolo(self.frame)[0].boxes.data

    def _get_scaling(self):
        """Calculate window scaling parameters to maintain aspect ratio"""
        assert self.frame is not None
        height, width = self.frame.shape[:2]
        win_w, win_h = self.width(), self.height()

        if (win_w / win_h) > (width / height):
            # Window is wider than image
            scaled_h = win_h
            scaled_w = int(win_h * width / height)
            offset_x = (win_w - scaled_w) // 2
            offset_y = 0
        else:
            # Window is taller than image
            scaled_w = win_w
            scaled_h = int(win_w * height / width)
            offset_x = 0
            offset_y = (win_h - scaled_h) // 2

        return WindowScaling(scaled_w, scaled_h, offset_x, offset_y, scaled_w / width, scaled_h / height)

    def paintEvent(self, event):
        if self.frame is not None:
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            self.scaling = self._get_scaling()

            # Draw camera image
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            qp = QPainter(self)
            qp.drawPixmap(
                QRect(self.scaling.x_offset, self.scaling.y_offset, self.scaling.scaled_width, self.scaling.scaled_height),
                QPixmap.fromImage(qimage).scaled(self.scaling.scaled_width, self.scaling.scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            if self.is_frozen:
                self.draw_masks(qp)
                self.draw_detections(qp)


            # Calculate and draw buttons
            button_y = self.scaling.y_offset + self.scaling.scaled_height - self.button_radius - 10
            self.freeze_button_center = (self.scaling.x_offset + self.scaling.scaled_width // 2, button_y)

            # Draw clear button if frozen
            if self.is_frozen:
                clear_width = 60
                clear_height = 30
                clear_x = self.scaling.x_offset + self.scaling.scaled_width - clear_width - 10
                clear_y = self.scaling.y_offset + self.scaling.scaled_height - clear_height - 10
                self.clear_button_rect = QRect(clear_x, clear_y, clear_width, clear_height)

                qp.setPen(QPen(QColor(255, 255, 255), 2))
                qp.setBrush(QColor(128, 128, 128, 180))
                qp.drawRect(self.clear_button_rect)
                qp.drawText(self.clear_button_rect, Qt.AlignCenter, "Clear")

            # Draw freeze/unfreeze button
            qp.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
            if not self.is_frozen:
                # Draw circle
                qp.setBrush(QColor(255, 255, 255, 128))
                qp.drawEllipse(
                    self.freeze_button_center[0] - self.button_radius,
                    self.freeze_button_center[1] - self.button_radius,
                    self.button_radius * 2,
                    self.button_radius * 2,
                )
            else:
                # Draw X
                qp.setBrush(Qt.NoBrush)
                x, y = self.freeze_button_center
                r = self.button_radius
                qp.drawLine(x - r, y - r, x + r, y + r)
                qp.drawLine(x - r, y + r, x + r, y - r)

    def draw_masks(self, qp=None):
        if not self.masks or self.frame is None: return
        if qp is None: qp = QPainter(self)
        height, width = self.frame.shape[:2]

        # Draw each mask with its color
        for mask in self.masks:
            mask_image = cv2.resize(mask.active_mask.astype(np.uint8) * 255, (width, height))
            mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
            r, g, b = mask.color.rgb
            mask_colored[mask_image > 0] = [r, g, b, 128]  # Semi-transparent color

            mask_qimage = QImage(mask_colored.data, width, height, 4 * width, QImage.Format_RGBA8888)
            qp.drawPixmap(
                QRect(self.scaling.x_offset, self.scaling.y_offset, self.scaling.scaled_width, self.scaling.scaled_height),
                QPixmap.fromImage(mask_qimage).scaled(self.scaling.scaled_width, self.scaling.scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Draw segmentation point with matching color
            x, y = mask.point
            _x = int(x * self.scaling.scale_x) + self.scaling.x_offset
            _y = int(y * self.scaling.scale_y) + self.scaling.y_offset
            qp.setPen(QPen(QColor(*mask.color.rgb), 4))
            qp.drawPoint(_x, _y)

    def draw_detections(self, qp):
        if self.detections is None: return

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

    def _window_to_image_coords(self, event):
        """Convert window coordinates to image coordinates"""
        assert self.frame is not None
        height, width = self.frame.shape[:2]
        scaling = self._get_scaling()

        # Convert click to image coordinates
        img_x = (event.x() - scaling.x_offset) / scaling.scale_x
        img_y = (event.y() - scaling.y_offset) / scaling.scale_y

        return Point(img_x, img_y), width, height

    def toggle_point_label(self, event):
        point, width, height = self._window_to_image_coords(event)
        if not ((0 <= point.x < width) and (0 <= point.y < height)): return
        for mask in self.masks:
            if abs(mask.point.x - point.x) <= SEG_POINT_RADIUS and abs(mask.point.y - point.y) <= SEG_POINT_RADIUS:
                mask.label = 1 if mask.label == 0 else 0
                print(mask.label)
                # self.draw_masks()
                self.segment(event)
                break

    def segment(self, event):
        point, width, height = self._window_to_image_coords(event)
        if not ((0 <= point.x < width) and (0 <= point.y < height)): return

        # check if click is a pre-existing point
        repeat_idx = None
        for idx, (x, y) in enumerate(self.seg_points):
            if abs(x - point.x) <= SEG_POINT_RADIUS and abs(y - point.y) <= SEG_POINT_RADIUS:
                repeat_idx = idx
                break

        if repeat_idx is not None:  # cycle mask
            mask = self.masks[repeat_idx]
            if mask.active < len(mask.masks) - 1:
                mask.active += 1
                mask.color = MaskColor.get_next_color(mask.color)
            else:
                self.masks.pop(repeat_idx)
        else:  # new mask
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam2.set_image(self.frame)
                masks, _, _ = sam2.predict([point], [1])
            print(f"Computed {len(masks)} masks for {point}")
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
