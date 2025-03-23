import sys
import cv2
import csv
import numpy as np
from skimage.filters import threshold_triangle
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPaintEvent, QImage, QPainter, QMouseEvent, QPen, QKeyEvent, QContextMenuEvent, QGuiApplication
from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction, QMainWindow, QDockWidget, QFormLayout, QSpinBox, QWidget, QHBoxLayout, QPushButton, QSlider, QFileDialog, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cell Thresholder App")
        self.setCentralWidget(FluoImage(self))

        # Get screen size
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        width = screen_geometry.width()
        height = screen_geometry.height()

        # Set fixed size to maximum available screen size
        self.setFixedSize(width - 50, height - 50)

class LabelCutOperation:
    def __init__(self):
        self.points = []
    
    def mousePressEvent(self, event: QMouseEvent):
        if len(self.points) == 2:
            self.points.clear()
        else:
            self.points.append((event.x(), event.y()))
    
    def paintEvent(self, painter: QPainter):
        if len(self.points) == 1:
            (x1, y1) = self.points[0]
            painter.drawPoint(x1, y1)
        elif len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    
    def applyTo(self, image, scale):
        if len(self.points) < 2:
            return image

        height, width = image.shape[:2]
        (x1, y1), (x2, y2) = self.points
        p1_scaled, p2_scaled = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, p1_scaled, p2_scaled, (1), thickness=-1)

        image[mask == 1] = np.average(image)
        return image

class CircleMaskOperation:
    def __init__(self):
        self.points = []
    
    def mousePressEvent(self, event: QMouseEvent):
        if len(self.points) == 3:
            self.points.clear()
        else:
            self.points.append((event.x(), event.y()))
    
    def paintEvent(self, painter: QPainter):
        if len(self.points) < 3:
            for (x, y) in self.points:
                painter.drawPoint(x, y)
        elif len(self.points) == 3:
            cx, cy, radius = self.calculate_circle()
            painter.drawEllipse(QPoint(int(cx), int(cy)), int(radius), int(radius))
    
    def applyTo(self, image, scale):
        if len(self.points) < 3:
            return

        height, width = image.shape[:2]
        cx, cy, radius = self.calculate_circle()
        cx_scaled, cy_scaled, radius_scaled = int(cx * scale), int(cy * scale), int(radius * scale)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx_scaled, cy_scaled), radius_scaled, (1), thickness=-1)

        image[mask == 0] = np.average(image)
        return image
    
    def calculate_circle(self):
        (x1, y1), (x2, y2), (x3, y3) = self.points

        D = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        
        cx = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / D
        cy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / D
        
        radius = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        return cx, cy, radius

class ThresholdWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, fluo_image, value):
        super().__init__()
        self.fluo_image = fluo_image
        self.value = value
    
    def run(self):
        if self.fluo_image.segmentation_mask is not None:
            ids = np.unique(self.fluo_image.segmentation_mask)[1:]

            if self.fluo_image.intensities is None:
                self.fluo_image.intensities = {}

                image, mask = self.fluo_image.image, self.fluo_image.segmentation_mask
                offset_x, offset_y = self.fluo_image.offset_x, self.fluo_image.offset_y
                cut_image = image[
                    max(offset_y, 0) : min(offset_y + mask.shape[0], image.shape[0]),
                    max(offset_x, 0) : min(offset_x + mask.shape[1], image.shape[1]),
                ]
                cut_mask = mask[
                    max(-offset_y, 0) : min(mask.shape[0], image.shape[0] - offset_y),
                    max(-offset_x, 0) : min(mask.shape[1], image.shape[1] - offset_x),
                ]

                for id in ids:
                    intensity = np.sum(cut_image[cut_mask == id]) / 255.0
                    self.fluo_image.intensities[id] = intensity
            
            processed_mask = np.copy(self.fluo_image.segmentation_mask)

            for id in ids:
                intensity = self.fluo_image.intensities[id]
                if intensity < self.value:
                    processed_mask[self.fluo_image.segmentation_mask == id] = 0
            
            self.fluo_image.thresholded_mask = processed_mask
            self.fluo_image.update()
        
        self.finished.emit()

class SegmentWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, fluo_image, gaussian_kernel_size, area_threshold):
        super().__init__()
        self.fluo_image = fluo_image
        self.gaussian_kernel_size = gaussian_kernel_size
        self.area_threshold = area_threshold
    
    def run(self):
        blurred = cv2.GaussianBlur(self.fluo_image.image, self.gaussian_kernel_size, 0)

        retval, binary_image = cv2.threshold(blurred, threshold_triangle(blurred) + np.std(blurred), 255, cv2.THRESH_BINARY)
        num_labels, labels_im = cv2.connectedComponents(binary_image)

        label_sizes = np.bincount(labels_im.ravel())
        invalid_labels = (label_sizes < self.area_threshold[0]) | (label_sizes > self.area_threshold[1])
        labels_im[np.isin(labels_im, np.where(invalid_labels)[0])] = 0

        self.fluo_image.segmentation_mask = labels_im
        self.fluo_image.thresholded_mask = labels_im
        self.fluo_image.intensities = None
        self.fluo_image.update()

        self.finished.emit()

class FluoImage(QLabel):
    def __init__(self, main_window: QMainWindow):
        super().__init__()

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.main_window = main_window
        self.image = None
        self.segmentation_mask = None
        self.thresholded_mask = None
        self.intensities = None
        self.ongoing_operation = None
        self.scale = None
        self.offset_x = 0
        self.offset_y = 0

        self.add_threshold_dock()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drop fluo image")
        self.setStyleSheet("border: 1px solid black")
    
    def paintEvent(self, event: QPaintEvent):
        if self.image is None:
            return super().paintEvent(event)

        painter = QPainter(self)
        pen = QPen(Qt.green, 5)
        painter.setPen(pen)
        
        height, width = self.image.shape
        qimage = QImage(self.image.data, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio)

        painter.drawPixmap(0, 0, scaled_pixmap)
        if self.ongoing_operation is not None:
            self.ongoing_operation.paintEvent(painter)
        if self.segmentation_mask is not None:
            self.overlay_mask(painter)

        painter.end()
    
    def overlay_mask(self, painter: QPainter):
        color_mask = cv2.applyColorMap(self.thresholded_mask.astype(np.uint8), cv2.COLORMAP_JET)

        height, width = color_mask.shape[:2]
        qimage_mask = QImage(color_mask.data, width, height, color_mask.strides[0], QImage.Format_RGB888)

        pixmap_mask = QPixmap.fromImage(qimage_mask)
        
        scaled_mask = pixmap_mask.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio)

        painter.setOpacity(0.3)
        painter.drawPixmap(self.offset_x, self.offset_y, scaled_mask)
        painter.setOpacity(1.0)

    def keyPressEvent(self, event: QKeyEvent):
        if self.ongoing_operation is not None and event.key() == 16777220:
            self.image = self.ongoing_operation.applyTo(self.image, self.scale)
            self.ongoing_operation = None
            self.update()
        elif event.key() == 16777216:
            self.ongoing_operation = None
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.ongoing_operation is not None: 
            self.ongoing_operation.mousePressEvent(event)
            self.update()
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.endswith("npz") and self.image is not None:
                seg = np.load(url)
                self.segmentation_mask = seg["im_markers"]
                self.thresholded_mask = seg["im_markers"]
                self.intensities = None
                self.update()
            else:
                self.image = cv2.imdecode(np.fromfile(url, np.uint8), cv2.IMREAD_GRAYSCALE)
                self.segmentation_mask = None
                self.thresholded_mask = None
                self.intensities = None
                self.ongoing_operation = None
                self.offset_x = 0
                self.offset_y = 0
                self.scale = max(self.image.shape[0] / self.height(), self.image.shape[1] / self.width())
                self.update()

    def contextMenuEvent(self, event: QContextMenuEvent):
        if self.image is not None and self.ongoing_operation is None:
            context_menu = QMenu(self)

            action1 = QAction("Circle mask", self)
            action1.triggered.connect(self.start_circle_mask_operation)
            context_menu.addAction(action1)

            action2 = QAction("Label cut", self)
            action2.triggered.connect(self.start_label_cut_operation)
            context_menu.addAction(action2)

            action3 = QAction("Segment", self)
            action3.triggered.connect(self.add_segmentation_dock)
            context_menu.addAction(action3)

            action4 = QAction("Export", self)
            action4.triggered.connect(self.start_exporting)
            context_menu.addAction(action4)

            context_menu.exec_(event.globalPos())
    
    def start_circle_mask_operation(self):
        self.ongoing_operation = CircleMaskOperation()
    
    def start_label_cut_operation(self):
        self.ongoing_operation = LabelCutOperation()
    
    def start_exporting(self):
        if self.thresholded_mask is None or self.intensities is None:
            self.show_error("Unfinished workflow", "Please segment and threshold the image first.")
            return

        cell_ids = np.unique(self.thresholded_mask)[1:]
        cell_count, areas, integrals = len(cell_ids), {}, {}
        for id in cell_ids:
            areas[id] = np.count_nonzero(self.thresholded_mask == id)
            integrals[id] = self.intensities[id]
        
        # Open file dialog to get the save path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if not file_path:
            return  # User canceled
        
        # Write data to CSV
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["Total Cell Count", cell_count])
            writer.writerow(["Cell ID", "Area", "Integral Intensity"])
            
            for id in cell_ids:
                writer.writerow([id, areas[id], integrals[id]])

    def add_segmentation_dock(self):
        dock_widget = QDockWidget("Segment", self)

        widget = QWidget()
        dock_widget.setWidget(widget)

        layout = QFormLayout()
        kernel_layout = QHBoxLayout()

        kernel_x = QSpinBox()
        kernel_x.setRange(1, 100)
        kernel_x.setValue(5)
        kernel_layout.addWidget(kernel_x)

        kernel_y = QSpinBox()
        kernel_y.setRange(1, 100)
        kernel_y.setValue(5)
        kernel_layout.addWidget(kernel_y)

        layout.addRow("Gaussian kernel size", kernel_layout)

        area_min_threshold = QSpinBox()
        area_min_threshold.setRange(0, 1000000)
        area_min_threshold.setValue(500)

        layout.addRow("Area min threshold", area_min_threshold)

        area_max_threshold = QSpinBox()
        area_max_threshold.setRange(0, 1000000)
        area_max_threshold.setValue(100000)

        layout.addRow("Area max threshold", area_max_threshold)

        apply_segmentation_button = QPushButton("Apply")
        layout.addRow(apply_segmentation_button)

        def start_segment_task():
            apply_segmentation_button.setEnabled(False)
            apply_segmentation_button.setText("Calculating...")

            self.worker_thread = QThread()
            self.worker = SegmentWorker(
                self,
                (kernel_x.value(), kernel_y.value()),
                (area_min_threshold.value(), area_max_threshold.value())
            )
            self.worker.moveToThread(self.worker_thread)
            
            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(lambda: apply_segmentation_button.setText("Apply"))
            self.worker.finished.connect(lambda: apply_segmentation_button.setEnabled(True))
            self.worker_thread.start()
        
        apply_segmentation_button.clicked.connect(start_segment_task)

        offset_layout = QHBoxLayout()

        offset_x = QSpinBox()
        offset_x.setRange(-5000, 5000)
        offset_x.setValue(0)
        offset_layout.addWidget(offset_x)

        offset_y = QSpinBox()
        offset_y.setRange(-5000, 5000)
        offset_y.setValue(0)
        offset_layout.addWidget(offset_y)

        layout.addRow("Offset", offset_layout)

        def apply_offset():
            self.offset_x = offset_x.value()
            self.offset_y = offset_y.value()
            self.intensities = None
            self.update()

        apply_offset_button = QPushButton("Apply")
        apply_offset_button.clicked.connect(apply_offset)
        layout.addRow(apply_offset_button)

        widget.setLayout(layout)

        self.main_window.addDockWidget(2, dock_widget)
    
    def add_threshold_dock(self):
        dock_widget = QDockWidget("Threshold", self)
        dock_widget.setFeatures(dock_widget.DockWidgetFeature.DockWidgetVerticalTitleBar)

        widget = QWidget()
        dock_widget.setWidget(widget)

        layout = QFormLayout()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 2000)
        layout.addRow("Value", slider)

        label = QLabel(f"{slider.value()}")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(label)
        slider.valueChanged.connect(lambda: label.setText(f"{slider.value()}"))

        apply_button = QPushButton("Apply")
        layout.addRow(apply_button)

        def start_threshold_task():
            apply_button.setEnabled(False)
            apply_button.setText("Calculating...")

            self.worker_thread = QThread()
            self.worker = ThresholdWorker(self, slider.value())
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(lambda: apply_button.setText("Apply"))
            self.worker.finished.connect(lambda: apply_button.setEnabled(True))
            self.worker_thread.start()

        apply_button.clicked.connect(start_threshold_task)

        widget.setLayout(layout)

        self.main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_widget)

    def show_error(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    app.exec()

if __name__ == "__main__":
    main()
