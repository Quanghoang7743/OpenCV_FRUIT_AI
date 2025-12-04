import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
from pathlib import Path
import os
from datetime import datetime
import threading, queue, time
from collections import Counter


class FruitDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Detector - Model Evaluation")
        self.root.geometry("1400x850")
        
        # Variables
        self.image_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.photo_image = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.camera_running = False
        self.cam_thread = None
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Configure customtkinter theme
        ctk.set_appearance_mode("dark")  # "light" or "dark"
        ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
        
        # Initialize UI
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        main_container = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header with gradient effect
        header_frame = ctk.CTkFrame(main_container, corner_radius=10, height=80)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)  
        
        # Header content
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.place(relx=0.5, rely=0.5, anchor="center")
        
        header_label = ctk.CTkLabel(
            header_content,
            text="FRUIT DETECTOR",
            font=("Segoe UI", 28, "bold")
        )
        header_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            header_content,
            text="AI-Powered Model Evaluation System",
            font=("Segoe UI", 13),
            text_color=("gray60", "gray50")
        )
        subtitle_label.pack()
        
        # Content container
        content_container = ctk.CTkFrame(main_container, corner_radius=0, fg_color="transparent")
        content_container.pack(fill="both", expand=True)
        
        # Left panel - Image display
        left_panel = ctk.CTkFrame(content_container, corner_radius=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        # Image title section
        image_header = ctk.CTkFrame(left_panel, corner_radius=8, height=45)
        image_header.pack(fill="x", padx=15, pady=(15, 10))
        image_header.pack_propagate(False)
        
        image_title = ctk.CTkLabel(
            image_header,
            text="Image Preview",
            font=("Segoe UI", 16, "bold")
        )
        image_title.place(relx=0.5, rely=0.5, anchor="center")
        
        # Image display frame
        self.image_frame = ctk.CTkFrame(left_panel, corner_radius=8, fg_color=("gray90", "gray20"))
        self.image_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="No image selected\n\nClick 'Choose Image' to get started",
            font=("Segoe UI", 14),
            text_color=("gray50", "gray60")
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Right panel - Controls and results
        right_panel = ctk.CTkFrame(content_container, corner_radius=10, width=450)
        right_panel.pack(side="right", fill="both", padx=(8, 0))
        right_panel.pack_propagate(False)
        
        # Status bar at top
        status_container = ctk.CTkFrame(right_panel, corner_radius=8, height=60)
        status_container.pack(fill="x", padx=15, pady=(15, 10))
        status_container.pack_propagate(False)
        
        status_left = ctk.CTkFrame(status_container, fg_color="transparent")
        status_left.pack(side="left", fill="y", padx=10)
        
        ctk.CTkLabel(
            status_left,
            text="Model Status",
            font=("Segoe UI", 11),
            text_color=("gray50", "gray60")
        ).pack(anchor="w")
        
        self.status_label = ctk.CTkLabel(
            status_left,
            text="Loading...",
            font=("Segoe UI", 13, "bold"),
            text_color=("#f39c12", "#f39c12")
        )
        self.status_label.pack(anchor="w")
        
        status_right = ctk.CTkFrame(status_container, fg_color="transparent")
        status_right.pack(side="right", fill="y", padx=10)
        
        ctk.CTkLabel(
            status_right,
            text="Device",
            font=("Segoe UI", 11),
            text_color=("gray50", "gray60")
        ).pack(anchor="e")
        
        device_text = "GPU" if self.device.type == "cuda" else "CPU"
        ctk.CTkLabel(
            status_right,
            text=device_text,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="e")
        
        # Control buttons section
        control_frame = ctk.CTkFrame(right_panel, corner_radius=8)
        control_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            control_frame,
            text="Actions",
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Choose image button
        self.choose_btn = ctk.CTkButton(
            control_frame,
            text="Choose Image",
            command=self.choose_image,
            font=("Segoe UI", 14, "bold"),
            height=45,
            corner_radius=8,
            hover_color=("#1f6aa5", "#144870")
        )
        self.choose_btn.pack(fill="x", pady=(0, 10), padx=15)
        
        # Camera button
        self.camera_btn = ctk.CTkButton(
            control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=("Segoe UI", 14, "bold"),
            height=45,
            corner_radius=8
        )
        self.camera_btn.pack(fill="x", pady=(0, 10), padx=15)
        
        # Test dataset button
        self.test_btn = ctk.CTkButton(
            control_frame,
            text="Test Dataset",
            command=self.test_dataset,
            font=("Segoe UI", 14, "bold"),
            height=45,
            corner_radius=8,
            fg_color=("#9b59b6", "#8e44ad"),
            hover_color=("#8e44ad", "#7d3c98")
        )
        self.test_btn.pack(fill="x", pady=(0, 15), padx=15)
        
        # Image info section
        info_frame = ctk.CTkFrame(right_panel, corner_radius=8)
        info_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text="Image Info",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(12, 8), padx=15, anchor="w")
        
        self.image_info_label = ctk.CTkLabel(
            info_frame,
            text="No image loaded",
            font=("Segoe UI", 11),
            text_color=("gray60", "gray50"),
            justify="left",
            anchor="w"
        )
        self.image_info_label.pack(fill="x", pady=(0, 12), padx=15, anchor="w")
        
        # Detection results section
        results_frame = ctk.CTkFrame(right_panel, corner_radius=8)
        results_frame.pack(fill="both", expand=True, padx=15, pady=(10, 15))
        
        ctk.CTkLabel(
            results_frame,
            text="Detection Results",
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(15, 10), padx=15, anchor="w")
        
        # Results text area with scrollbar
        results_container = ctk.CTkFrame(results_frame, fg_color="transparent")
        results_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.results_text = ctk.CTkTextbox(
            results_container,
            font=("Consolas", 11),
            corner_radius=6,
            wrap="word"
        )
        self.results_text.pack(fill="both", expand=True)
        

        ##### LOGIC 
    def load_model(self):
        try:
            from ultralytics import YOLO
            
            model_path = "fruit_detector_last.pt"
            if not os.path.exists(model_path):
                self.status_label.configure(text="Model not found", text_color="#e74c3c")
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return False
            
            self.status_label.configure(text="Loading model...", text_color="#f39c12")
            self.root.update()
            
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            self.status_label.configure(text="Ready", text_color="#2ecc71")
            return True
            
        except Exception as e:
            self.status_label.configure(text="Error", text_color="#e74c3c")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return False
    
    def choose_image(self):
        """Choose image file"""
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        )
        
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )
        
        if filename:
            self.image_path = filename
            self.display_image()
            self.detect_fruits()
    
    def display_image(self):
        """Display selected image"""
        if not self.image_path:
            return
        
        try:
            # Read image
            img = Image.open(self.image_path)
            
            # Resize for display
            max_size = (600, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.configure(image=self.photo_image, text="")
            
            # Update image info
            file_name = os.path.basename(self.image_path)
            file_size = os.path.getsize(self.image_path) / 1024  # KB
            img_size = Image.open(self.image_path).size
            
            info_text = f"{file_name}\n {file_size:.1f} KB\n {img_size[0]} x {img_size[1]} px"
            self.image_info_label.configure(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\n{str(e)}")
    
    def display_image_with_detections(self, img_cv2, detections):
        try:
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Resize for display
            max_size = (480, 480)
            img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(img_pil)
            
            # Update label
            self.image_label.configure(image=self.photo_image, text="")
            
        except Exception as e:
            print(f"Error displaying image with detections: {e}")
    
    def detect_fruits(self):
        """Detect fruits in image"""
        if not self.image_path or not self.model:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.choose_btn.configure(state="disabled")
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", "Processing...\n\n")
            self.root.update()
            
            # Read image
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("Failed to read image")
            
            # Run detection
            results = self.model.predict(img, verbose=False)
            
            # Process results
            self.results_text.delete("1.0", "end")
            
            result = results[0]
            detections = result.boxes
            
            # Draw detections on image
            img_with_boxes = img.copy()
            
            # Color palette for different classes
            colors = [
                (0, 255, 0),      # Green
                (255, 0, 0),      # Blue
                (0, 0, 255),      # Red
                (255, 255, 0),    # Cyan
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Yellow
                (128, 0, 255),    # Orange
                (255, 128, 0),    # Purple
                (0, 128, 255),    # Red-Orange
            ]
            
            # Header
            header = f"{'='*38}\n"
            header += f"{'DETECTION RESULTS':^38}\n"
            header += f"{'='*38}\n\n"
            self.results_text.insert("end", header)
            
            if detections is None or len(detections) == 0:
                self.results_text.insert("end", "No fruits detected\n")
                self.display_image_with_detections(img_with_boxes, detections)
            else:
                # Summary
                summary = f"Total Detections: {len(detections)}\n\n"
                self.results_text.insert("end", summary)
                
                # Details
                self.results_text.insert("end", f"{'#':<3} {'Class':<15} {'Confidence':<12} {'Box':<10}\n")
                self.results_text.insert("end", "-" * 38 + "\n")
                
                class_counts = {}
                
                for idx, det in enumerate(detections, 1):
                    cls = int(det.cls.item()) if hasattr(det.cls, 'item') else int(det.cls)
                    conf = float(det.conf.item()) if hasattr(det.conf, 'item') else float(det.conf)
                    
                    # Get class name
                    class_name = self.model.names.get(cls, f"Class_{cls}") if hasattr(self.model, 'names') else f"Class_{cls}"
                    
                    # Count classes
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Get box coordinates and draw
                    if hasattr(det, 'xyxy'):
                        x1, y1, x2, y2 = det.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Choose color based on class
                        color = colors[cls % len(colors)]
                        
                        # Draw rectangle
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                        
                        # Prepare label
                        label = f"{class_name} {conf:.2%}"
                        
                        # Get text size
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 1
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img_with_boxes, 
                                    (x1, y1 - text_size[1] - 4),
                                    (x1 + text_size[0] + 4, y1),
                                    color, -1)
                        
                        # Put text
                        cv2.putText(img_with_boxes, label, 
                                  (x1 + 2, y1 - 2),
                                  font, font_scale, (255, 255, 255), thickness)
                        
                        box_str = "Yes"
                    else:
                        box_str = "No"
                    
                    # Format and display
                    conf_str = f"{conf:.2%}"
                    line = f"{idx:<3} {class_name:<15} {conf_str:<12} {box_str:<10}\n"
                    self.results_text.insert("end", line)
                
                # Class summary
                self.results_text.insert("end", "\n" + "-" * 38 + "\n")
                self.results_text.insert("end", "CLASS SUMMARY:\n")
                for class_name in sorted(class_counts.keys()):
                    count = class_counts[class_name]
                    self.results_text.insert("end", f"  • {class_name}: {count}\n")
                
                # Statistics
                confidences = [float(det.conf.item() if hasattr(det.conf, 'item') else det.conf) for det in detections]
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
                
                self.results_text.insert("end", f"\nSTATISTICS:\n")
                self.results_text.insert("end", f"  Avg Confidence: {avg_conf*100:.2f}%\n")
                self.results_text.insert("end", f"  Min Confidence: {min_conf*100:.2f}%\n")
                self.results_text.insert("end", f"  Max Confidence: {max_conf*100:.2f}%\n")
                
                # Display image with boxes
                self.display_image_with_detections(img_with_boxes, detections)
            
            # Inference time
            if hasattr(result, 'speed'):
                inference_time = result.speed.get('inference', 0) if isinstance(result.speed, dict) else 0
                self.results_text.insert("end", f"\n⏱Inference: {inference_time:.1f}ms\n")
            
            self.results_text.insert("end", "\n" + "=" * 38 + "\n")
            self.choose_btn.configure(state="normal")
            
        except Exception as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", f"Error: {str(e)}\n")
            self.choose_btn.configure(state="normal")
    
    def test_dataset(self):
        """Test on dataset images"""
        dataset_path = "Fruits_data/test/images"
        
        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", f"Dataset path not found: {dataset_path}")
            return
        
        # Get random image from dataset
        images = list(Path(dataset_path).glob("*.jpg"))
        if not images:
            messagebox.showerror("Error", "No images found in dataset")
            return
        
        import random
        random_image = random.choice(images)
        
        self.image_path = str(random_image)
        self.display_image()
        self.detect_fruits()
        
        messagebox.showinfo("Info", f"Tested with random image:\n{random_image.name}")


    def toggle_camera(self):
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if self.model is None:
            if not self.load_model():
                return

        import sys
        backends = []
        if sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        indices = [1, 0]

        opened = False
        last_err = None
        for be in backends:
            for idx in indices:
                try:
                    cap = cv2.VideoCapture(idx, be)
                    if not (cap and cap.isOpened()):
                        if cap: cap.release()
                        continue

                    # Set size vừa phải để mượt (có thể đổi 1280x720 tuỳ máy)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                    # WARM-UP: bỏ 5 frame đầu để sensor/driver ổn định
                    ok = True
                    for _ in range(5):
                        ret, _ = cap.read()
                        if not ret:
                            ok = False
                            break
                    if not ok:
                        cap.release()
                        continue

                    # Thử đọc 1 frame “chính thức”
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.release()
                        continue

                    # Nếu tới đây là ngon
                    self.cap = cap
                    opened = True
                    self.results_text.delete("1.0", "end")
                    self.results_text.insert("end", f"Camera OK → backend={be}, index={idx}\n")
                    break

                except Exception as e:
                    last_err = e
            if opened:
                break

        if not opened:
            msg = "Không mở được camera.\n"
            if last_err:
                msg += f"Chi tiết: {last_err}\n"
            msg += (
                "Gợi ý:\n"
                "• System Settings → Privacy & Security → Camera: bật quyền cho app bạn dùng để chạy (Terminal/VS Code).\n"
                "• Đóng Zoom/FaceTime/Meet…\n"
                "• Dùng index 1 thay vì 0 (theo test bạn), hoặc rút/cắm lại webcam.\n"
            )
            messagebox.showerror("Camera Error", msg)
            self.stop_camera()
            return

        self.camera_running = True
        self.choose_btn.configure(state="disabled")
        self.camera_btn.configure(text="Stop Camera")
        self.status_label.configure(text="Live", text_color="#2ecc71")

        self.results_text.insert("end", "LIVE MODE: Đưa quả trước camera để nhận diện.\n")

        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()
        self.root.after(15, self._update_camera_view)


    def stop_camera(self):
        """Gracefully stop webcam and thread."""
        self.camera_running = False
        try:
            if self.cam_thread and self.cam_thread.is_alive():
                self.cam_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.cap:
                self.cap.release()
        finally:
            self.cap = None

        self.camera_btn.configure(text="Start Camera")
        self.choose_btn.configure(state="normal")
        # Giữ status là Ready khi dừng live
        self.status_label.configure(text="Ready", text_color="#2ecc71")

    def _camera_worker(self):
        """Read frames, run YOLO, push annotated frames + summary to queue."""
        last_t = time.time()
        while self.camera_running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                continue

            t0 = time.time()
            # Tăng/giảm imgsz hoặc conf cho tốc độ/độ chính xác
            results = self.model.predict(frame, imgsz=640, conf=0.35, verbose=False)
            result = results[0]

            # Ảnh đã vẽ bbox (Ultralytics trả về BGR phù hợp OpenCV)
            annotated = result.plot()

            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - last_t))
            last_t = t1

            # Tóm tắt detections
            detections = result.boxes
            total = int(len(detections)) if detections is not None else 0

            class_counts = Counter()
            top_det_text = "—"

            if detections is not None and len(detections) > 0:
                # cls & conf là tensor -> list
                classes = [int(c) for c in detections.cls.detach().cpu().tolist()]
                confs   = [float(c) for c in detections.conf.detach().cpu().tolist()]
                names   = [self.model.names.get(c, f"Class_{c}") if hasattr(self.model, "names") else f"Class_{c}" for c in classes]
                for n in names:
                    class_counts[n] += 1
                # top theo confidence
                top_i = max(range(len(confs)), key=lambda i: confs[i])
                top_det_text = f"{names[top_i]} {confs[top_i]*100:.1f}%"

            # Build summary text gọn cho UI
            summary_lines = [
                "LIVE MODE (Stop Camera để quay lại chế độ ảnh tĩnh)",
                f"FPS: {fps:.1f} | Tổng phát hiện: {total}",
            ]
            if class_counts:
                summary_lines.append("Tổng theo lớp:")
                for k, v in class_counts.items():
                    summary_lines.append(f"  • {k}: {v}")
                summary_lines.append(f"Top: {top_det_text}")
            else:
                summary_lines.append("Chưa thấy quả nào trong khung hình.")

            summary = "\n".join(summary_lines)

            # Đẩy frame + summary vào queue (giữ queue kích thước 1 để không backlog)
            try:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put_nowait((annotated, summary))
            except queue.Full:
                pass

    def _update_camera_view(self):
        """Pull frame from queue and render to Tk; reschedule if running."""
        if not self.camera_running:
            return
        try:
            frame, summary = self.frame_queue.get_nowait()
            # BGR -> RGB cho PIL
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Resize vừa khung hiển thị
            max_size = (480, 480)
            img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(img_pil)
            self.image_label.configure(image=self.photo_image, text="")

            # Cập nhật text kết quả
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", summary)

        except queue.Empty:
            pass
        finally:
            # 10–20ms là ổn cho realtime UI
            self.root.after(20, self._update_camera_view)

    def on_close(self):
        """Đảm bảo giải phóng camera trước khi thoát."""
        try:
            self.stop_camera()
        finally:
            self.root.destroy()

def main():
    root = ctk.CTk()
    app = FruitDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
