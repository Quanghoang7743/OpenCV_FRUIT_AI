import cv2
import torch
import os
import threading
import queue
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import numpy as np
from core.statistics_manager import StatisticsManager

class FruitDetectorLogic:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cap = None
        self.camera_running = False
        self.cam_thread = None
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Statistics Manager
        self.statistics_manager = StatisticsManager()
        
        # Bảng màu cho bounding box
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

    def get_device_name(self):
        return "GPU" if self.device.type == "cuda" else "CPU"

    def load_model(self, model_path=None):
        try:
            # Ưu tiên đường dẫn cung cấp; mặc định trỏ tới models/fruit_detector_last.pt ở project root
            if model_path is None:
                project_root = Path(__file__).resolve().parent.parent.parent
                model_path = project_root / "models" / "fruit_detector_last.pt"
            else:
                model_path = Path(model_path)

            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            self.model = YOLO(str(model_path))
            self.model.to(self.device)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, str(e)

    def process_static_image(self, image_path):
        """Xử lý ảnh tĩnh và trả về ảnh đã vẽ box + chuỗi kết quả"""
        if not self.model:
            return None, "Model not loaded"

        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Failed to read image"

            results = self.model.predict(img, verbose=False)
            result = results[0]
            detections = result.boxes
            
            # Ghi nhận vào statistics
            if detections is not None and len(detections) > 0:
                self.statistics_manager.add_detection(detections, model_names=self.model.names)
            
            img_with_boxes = img.copy()
            output_text = self._format_static_results(detections, img_with_boxes)
            
            return img_with_boxes, output_text
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def _format_static_results(self, detections, img_draw):
        """Hàm phụ trợ để vẽ box và tạo báo cáo text"""
        header = f"{'='*38}\n{'DETECTION RESULTS':^38}\n{'='*38}\n\n"
        
        if detections is None or len(detections) == 0:
            return header + "No fruits detected\n"

        summary = header + f"Total Detections: {len(detections)}\n\n"
        summary += f"{'#':<3} {'Class':<15} {'Confidence':<12}\n{'-'*38}\n"
        
        class_counts = {}
        confidences = []

        for idx, det in enumerate(detections, 1):
            cls = int(det.cls.item()) if hasattr(det.cls, 'item') else int(det.cls)
            conf = float(det.conf.item()) if hasattr(det.conf, 'item') else float(det.conf)
            class_name = self.model.names.get(cls, f"Class_{cls}")
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(conf)

            # Vẽ Box
            if hasattr(det, 'xyxy'):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                color = self.colors[cls % len(self.colors)]
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2%}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(img_draw, (x1, y1 - t_size[1] - 4), (x1 + t_size[0] + 4, y1), color, -1)
                cv2.putText(img_draw, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            summary += f"{idx:<3} {class_name:<15} {conf:.2%}\n"

        # Thống kê
        summary += "\n" + "-" * 38 + "\nCLASS SUMMARY:\n"
        for name, count in sorted(class_counts.items()):
            summary += f"  • {name}: {count}\n"
            
        if confidences:
            summary += f"\nSTATISTICS:\n  Avg Conf: {sum(confidences)/len(confidences):.1%}\n"
            summary += f"  Max Conf: {max(confidences):.1%}\n"

        return summary

    def start_camera_stream(self):
        """Khởi tạo camera và thread"""
        if self.camera_running: 
            return True, "Camera already running"

        import sys
        backends = []
        if sys.platform == "darwin": backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif sys.platform.startswith("win"): backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else: backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        opened = False
        for be in backends:
            for idx in [1, 0]: # Thử index 1 trước
                try:
                    cap = cv2.VideoCapture(idx, be)
                    if cap and cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        # Warmup
                        for _ in range(5): cap.read()
                        self.cap = cap
                        opened = True
                        break
                except: continue
            if opened: break
        
        if not opened:
            return False, "Could not open camera"

        self.camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()
        return True, "Camera started"

    def stop_camera_stream(self):
        self.camera_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _camera_worker(self):
        last_t = time.time()
        while self.camera_running and self.cap:
            ok, frame = self.cap.read()
            if not ok: continue

            # Predict
            results = self.model.predict(frame, imgsz=640, conf=0.35, verbose=False)
            result = results[0]
            annotated = result.plot() # Ultralytics plot helper

            # FPS Calculation
            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - last_t))
            last_t = t1

            # Summary Text
            detections = result.boxes
            count = len(detections) if detections else 0
            summary = f"LIVE MODE\nFPS: {fps:.1f} | Objects: {count}\n"
            
            if count > 0:
                cls_list = [int(c) for c in detections.cls.cpu().tolist()]
                names = [self.model.names[c] for c in cls_list]
                counts = Counter(names)
                for k, v in counts.items():
                    summary += f"• {k}: {v}\n"
                
                # Ghi nhận vào statistics (mỗi 30 frame để tránh quá tải)
                if int(time.time() * 10) % 30 == 0:  # Sample every ~3 seconds at ~10 FPS
                    self.statistics_manager.add_detection(detections, model_names=self.model.names)

            # Put to queue
            try:
                if not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((annotated, summary))
            except: pass