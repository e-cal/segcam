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
    BLUE = (100, 150, 255)
    PINK = (233, 30, 99)
    YELLOW = (252, 190, 73)

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
    points: list[Point]  # click coordinates for all points in this mask
    labels: list[Literal[0, 1]]  # background (0) or foreground (1) for each point
    masks: np.ndarray  # computed masks for all points
    active: int  # index of the active mask variant
    color: MaskColor = MaskColor.BLUE
    name: str = ""  # display name for the mask

    def __init__(self, name: str):
        self.points = []
        self.labels = []
        self.masks = np.array([])
        self.active = 0
        self.color = MaskColor.BLUE
        self.name = name

    @property
    def active_mask(self):
        return self.masks[self.active] if len(self.masks) > 0 else None

    def add_point(self, point: Point, label: Literal[0, 1] = 1):
        self.points.append(point)
        self.labels.append(label)

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
        self.show_detections = True  # Add flag for detection visibility
        self.masks: list[Mask] = []
        self.selected_mask_index: int | None = None
        self.mask_button_width = 80
        self.mask_button_height = 30
        self.mask_button_spacing = 10

    @property
    def active_mask(self) -> Mask | None:
        if self.selected_mask_index is None:
            return None
        return self.masks[self.selected_mask_index]

    def initUI(self):
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('SegCam')
        self.show()

    def mousePressEvent(self, event):
        if not self.is_frozen:
            if self.is_freeze_button_press(event.x(), event.y()):
                self.freeze()
                self.is_frozen = True
                self.update()
            return

        if self.frame is None:
            return

        if self.is_freeze_button_press(event.x(), event.y()):
            self.is_frozen = False
            self.update()
        elif self.is_clear_button_press(event.x(), event.y()):
            self.masks.clear()
            self.update()
        elif self.is_show_hide_button_press(event.x(), event.y()):
            self.show_detections = not self.show_detections
            self.update()
        else:
            # Check all button areas first
            if (self.is_freeze_button_press(event.x(), event.y()) or
                self.is_clear_button_press(event.x(), event.y()) or
                self.is_show_hide_button_press(event.x(), event.y())):
                return

            # Check if click is on mask selection buttons
            mask_clicked, is_delete = self.get_clicked_mask_button(event.x(), event.y())
            if mask_clicked is not None:
                if is_delete:
                    # Delete the mask and update names/indices
                    self.masks.pop(mask_clicked)
                    # Update mask names
                    for i, mask in enumerate(self.masks):
                        mask.name = f"Mask {i + 1}"
                        mask.color = list(MaskColor)[i % len(MaskColor)]
                    # Update selected index
                    if self.selected_mask_index == mask_clicked:
                        self.selected_mask_index = None
                    elif self.selected_mask_index > mask_clicked:
                        self.selected_mask_index -= 1
                elif mask_clicked == len(self.masks):  # "Add Mask" button
                    mask_index = len(self.masks)
                    new_mask = Mask(f"Mask {mask_index + 1}")
                    # Set color based on mask index
                    colors = list(MaskColor)
                    new_mask.color = colors[mask_index % len(colors)]
                    self.masks.append(new_mask)
                    self.selected_mask_index = len(self.masks) - 1
                else:
                    self.selected_mask_index = mask_clicked
                self.update()
                return

            # Handle image clicks only if a mask is selected and not on any buttons
            if self.selected_mask_index is not None:
                if event.button() == Qt.LeftButton:
                    self.segment(event)
                else:
                    self.toggle_point_label(event)

    def is_freeze_button_press(self, x, y):
        assert self.freeze_button_center is not None
        distance = math.sqrt((x - self.freeze_button_center[0])**2 + (y - self.freeze_button_center[1])**2)
        return distance <= self.button_radius

    def is_clear_button_press(self, x, y):
        if not self.is_frozen or self.clear_button_rect is None:
            return False
        return self.clear_button_rect.contains(x, y)

    def is_show_hide_button_press(self, x, y):
        if not self.is_frozen or not hasattr(self, 'show_hide_button_rect'):
            return False
        return self.show_hide_button_rect.contains(x, y)

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
                self.draw_mask_buttons(qp)
                self.draw_masks(qp, draw_buttons=False)
                self.draw_detections(qp)


            # Calculate and draw buttons
            button_y = self.scaling.y_offset + self.scaling.scaled_height - self.button_radius - 10
            self.freeze_button_center = (self.scaling.x_offset + self.scaling.scaled_width // 2, button_y)

            # Draw clear button if frozen
            if self.is_frozen:
                button_width = 60
                button_height = 30
                button_spacing = 10
                
                # Clear button
                clear_x = self.scaling.x_offset + self.scaling.scaled_width - button_width - 10
                clear_y = self.scaling.y_offset + self.scaling.scaled_height - button_height - 10
                self.clear_button_rect = QRect(clear_x, clear_y, button_width, button_height)

                qp.setPen(QPen(QColor(255, 255, 255), 2))
                qp.setBrush(QColor(128, 128, 128, 180))
                qp.drawRect(self.clear_button_rect)
                qp.drawText(self.clear_button_rect, Qt.AlignCenter, "Clear")

                # Show/Hide button
                show_hide_x = clear_x
                show_hide_y = clear_y - button_height - button_spacing
                self.show_hide_button_rect = QRect(show_hide_x, show_hide_y, button_width, button_height)

                qp.setPen(QPen(QColor(255, 255, 255), 2))
                qp.setBrush(QColor(128, 128, 128, 180))
                qp.drawRect(self.show_hide_button_rect)
                qp.drawText(self.show_hide_button_rect, Qt.AlignCenter, "Hide" if self.show_detections else "Show")

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

    def get_clicked_mask_button(self, x: int, y: int) -> tuple[int | None, bool]:
        """Returns (mask_index, is_delete_button) or (None, False) if no button clicked"""
        window_height = self.height()
        total_height = (len(self.masks) + 1) * (self.mask_button_height + self.mask_button_spacing)
        start_y = (window_height - total_height) // 2

        for i in range(len(self.masks) + 1):
            button_y = start_y + i * (self.mask_button_height + self.mask_button_spacing)
            
            # Check main mask button
            if (0 <= x <= self.mask_button_width and 
                button_y <= y <= button_y + self.mask_button_height):
                
                # Check if X button was clicked (for existing masks only)
                if i < len(self.masks):
                    x_button_x = self.mask_button_width - 15
                    if x_button_x <= x <= self.mask_button_width and button_y <= y <= button_y + 20:
                        return i, True
                return i, False
            
        return None, False

    def draw_mask_buttons(self, qp: QPainter):
        """Draw mask selection buttons on the left side"""
        window_height = self.height()
        total_height = (len(self.masks) + 1) * (self.mask_button_height + self.mask_button_spacing)
        start_y = (window_height - total_height) // 2

        for i, mask in enumerate(self.masks):
            button_y = start_y + i * (self.mask_button_height + self.mask_button_spacing)
            
            # Main mask button
            button_rect = QRect(0, button_y, self.mask_button_width, self.mask_button_height)
            qp.setBrush(QColor(100, 100, 100, 180) if i == self.selected_mask_index else QColor(60, 60, 60, 180))
            
            qp.setPen(QPen(QColor(255, 255, 255), 2))
            qp.drawRect(button_rect)
            qp.drawText(button_rect, Qt.AlignCenter, mask.name)

            # Draw X button in top-right corner
            x_button_x = self.mask_button_width - 15
            qp.setPen(QPen(QColor(255, 100, 100), 2))
            qp.drawLine(x_button_x, button_y + 5, x_button_x + 10, button_y + 15)
            qp.drawLine(x_button_x, button_y + 15, x_button_x + 10, button_y + 5)

        # Draw "Add Mask" button
        add_button_y = start_y + len(self.masks) * (self.mask_button_height + self.mask_button_spacing)
        add_button_rect = QRect(0, add_button_y, self.mask_button_width, self.mask_button_height)
        qp.setBrush(QColor(60, 60, 60, 180))
        qp.setPen(QPen(QColor(255, 255, 255), 2))
        qp.drawRect(add_button_rect)
        qp.drawText(add_button_rect, Qt.AlignCenter, "Add Mask")

    def draw_masks(self, qp=None, draw_buttons=True):
        if not self.masks or self.frame is None: return
        if qp is None: qp = QPainter(self)
        height, width = self.frame.shape[:2]

        # Draw mask buttons if requested
        if draw_buttons:
            self.draw_mask_buttons(qp)

        # Draw each mask with its color
        for mask in self.masks:
            if mask.active_mask is None: continue
            mask_image = cv2.resize(mask.active_mask.astype(np.uint8) * 255, (width, height))
            mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
            r, g, b = mask.color.rgb
            mask_colored[mask_image > 0] = [r, g, b, 128]  # Semi-transparent color

            mask_qimage = QImage(mask_colored.data, width, height, 4 * width, QImage.Format_RGBA8888)
            qp.drawPixmap(
                QRect(self.scaling.x_offset, self.scaling.y_offset, self.scaling.scaled_width, self.scaling.scaled_height),
                QPixmap.fromImage(mask_qimage).scaled(self.scaling.scaled_width, self.scaling.scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # Only draw segmentation points for the selected mask
            if mask == self.active_mask:
                for point, label in zip(mask.points, mask.labels):
                    x, y = point
                    _x = int(x * self.scaling.scale_x) + self.scaling.x_offset
                    _y = int(y * self.scaling.scale_y) + self.scaling.y_offset
                    # Green for foreground (1), red for background (0)
                    color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
                    qp.setPen(QPen(color, 5))
                    qp.drawPoint(_x, _y)

    def draw_detections(self, qp):
        if self.detections is None or not self.show_detections: return

        qp.setPen(QPen(QColor(255, 0, 0), 2))
        qp.setBrush(Qt.NoBrush)
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
        if self.active_mask is None: return
        point, width, height = self._window_to_image_coords(event)
        if not ((0 <= point.x < width) and (0 <= point.y < height)): return

        # Find if we clicked near any existing point
        for i, existing_point in enumerate(self.active_mask.points):
            if abs(existing_point.x - point.x) <= SEG_POINT_RADIUS and abs(existing_point.y - point.y) <= SEG_POINT_RADIUS:
                # Toggle label for this point
                self.active_mask.labels[i] = 1 if self.active_mask.labels[i] == 0 else 0
                
                # Recompute masks with updated labels
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sam2.set_image(self.frame)
                    masks, _, _ = sam2.predict(
                        self.active_mask.points,
                        self.active_mask.labels
                    )
                self.active_mask.masks = masks
                self.update()
                return

        # If no existing point was clicked, add a new negative point
        self.active_mask.points.append(point)
        self.active_mask.labels.append(0)  # Add as background point

        # Compute new masks with all points
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2.set_image(self.frame)
            masks, _, _ = sam2.predict(
                self.active_mask.points,
                self.active_mask.labels
            )
        self.active_mask.masks = masks
        self.active_mask.active = 0
        self.update()

    def segment(self, event):
        if self.active_mask is None: return
        point, width, height = self._window_to_image_coords(event)
        if not ((0 <= point.x < width) and (0 <= point.y < height)): return

        # Check if click is on existing point
        for i, existing_point in enumerate(self.active_mask.points):
            if abs(existing_point.x - point.x) <= SEG_POINT_RADIUS and abs(existing_point.y - point.y) <= SEG_POINT_RADIUS:
                # Remove the point
                self.active_mask.points.pop(i)
                self.active_mask.labels.pop(i)
                
                # Recompute masks if there are still points
                if self.active_mask.points:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        sam2.set_image(self.frame)
                        masks, _, _ = sam2.predict(
                            self.active_mask.points,
                            self.active_mask.labels
                        )
                    self.active_mask.masks = masks
                    self.active_mask.active = 0
                else:
                    self.active_mask.masks = np.array([])
                self.update()
                return

        # Add new point to active mask
        self.active_mask.points.append(point)
        # Right click adds background point (0), left click adds foreground point (1)
        self.active_mask.labels.append(1 if event.button() == Qt.LeftButton else 0)

        # Compute new masks with all points
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sam2.set_image(self.frame)
            masks, _, _ = sam2.predict(
                self.active_mask.points,
                self.active_mask.labels
            )
        self.active_mask.masks = masks
        self.active_mask.active = 0
        self.update()

    def wheelEvent(self, event):
        if self.active_mask and len(self.active_mask.masks) > 0:
            # Scroll up cycles forward, scroll down cycles backward
            delta = 1 if event.angleDelta().y() > 0 else -1
            self.active_mask.active = (self.active_mask.active + delta) % len(self.active_mask.masks)
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
