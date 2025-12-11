import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import queue
from pathlib import Path
from core.fruit_logic import FruitDetectorLogic 
from gui.dashboard import StatisticsDashboard

class FruitDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện trái cây")
        self.root.geometry("1400x850")
        
        # Init Logic
        self.logic = FruitDetectorLogic()
        
        # Variables
        self.image_path = None
        self.photo_image = None
        
        # Configure theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.setup_ui()
        self.root.after(100, self.init_model) # Load model after UI shows up
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """Thiết lập UI với tab view cho Detection và Dashboard"""
        
        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabview for Detection and Dashboard
        self.main_tabview = ctk.CTkTabview(main_container)
        self.main_tabview.pack(fill="both", expand=True)
        
        # Detection Tab
        detection_tab = self.main_tabview.add("Detection")
        self.setup_detection_tab(detection_tab)
        
        # Dashboard Tab
        dashboard_tab = self.main_tabview.add("Dashboard")
        self.setup_dashboard_tab(dashboard_tab)
    
    def setup_detection_tab(self, parent):
        """Thiết lập tab Detection"""
        # Left: Image
        left_panel = ctk.CTkFrame(parent)
        left_panel.pack(side="left", fill="both", expand=True, padx=5)
        
        self.image_label = ctk.CTkLabel(left_panel, text="No Image", width=600, height=600)
        self.image_label.pack(expand=True)

        # Right: Controls
        right_panel = ctk.CTkFrame(parent, width=400)
        right_panel.pack(side="right", fill="y", padx=5)
        
        self.status_label = ctk.CTkLabel(right_panel, text="Status: Init", text_color="orange")
        self.status_label.pack(pady=10)

        self.choose_btn = ctk.CTkButton(right_panel, text="Choose Image", command=self.choose_image)
        self.choose_btn.pack(pady=10, padx=20, fill="x")
        
        self.camera_btn = ctk.CTkButton(right_panel, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.pack(pady=10, padx=20, fill="x")
        
        self.results_text = ctk.CTkTextbox(right_panel, width=300)
        self.results_text.pack(pady=10, padx=20, fill="both", expand=True)
    
    def setup_dashboard_tab(self, parent):
        """Thiết lập tab Dashboard"""
        self.dashboard = StatisticsDashboard(parent, self.logic.statistics_manager)
        self.dashboard.pack(fill="both", expand=True)

    def init_model(self):
        self.status_label.configure(text="Loading Model...")
        self.root.update()
        success, msg = self.logic.load_model()
        if success:
            device = self.logic.get_device_name()
            self.status_label.configure(text=f"Ready ({device})", text_color="#2ecc71")
        else:
            self.status_label.configure(text="Error Loading Model", text_color="red")
            messagebox.showerror("Error", msg)

    def choose_image(self):
        filetypes = (("Images", "*.jpg *.png *.webp"), ("All", "*.*"))
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.image_path = path
            self.run_detection_static()

    def display_cv2_image(self, cv2_img):
        # Convert BGR (OpenCV) -> RGB (PIL)
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize giữ tỷ lệ
        w, h = img_pil.size
        aspect = w / h
        target_w = 600
        target_h = int(target_w / aspect)
        
        if target_h > 600:
            target_h = 600
            target_w = int(target_h * aspect)
            
        img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        self.image_label.configure(image=self.photo_image, text="")

    def run_detection_static(self):
        if not self.image_path: return
        
        self.status_label.configure(text="Processing...", text_color="yellow")
        self.root.update()
        
        # Gọi Logic
        img_result, text_result = self.logic.process_static_image(self.image_path)
        
        if img_result is not None:
            self.display_cv2_image(img_result)
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", text_result)
            self.status_label.configure(text="Done", text_color="#2ecc71")
            
            # Refresh dashboard if it exists
            if hasattr(self, 'dashboard'):
                self.dashboard.refresh_all()
        else:
            messagebox.showerror("Error", text_result)
            self.status_label.configure(text="Error", text_color="red")

    def toggle_camera(self):
        if not self.logic.camera_running:
            success, msg = self.logic.start_camera_stream()
            if success:
                self.camera_btn.configure(text="Stop Camera", fg_color="red")
                self.choose_btn.configure(state="disabled")
                self._update_camera_view()
            else:
                messagebox.showerror("Camera Error", msg)
        else:
            self.logic.stop_camera_stream()
            self.camera_btn.configure(text="Start Camera", fg_color=("#3a7ebf", "#1f538d"))
            self.choose_btn.configure(state="normal")

    def _update_camera_view(self):
        if not self.logic.camera_running:
            return

        try:
            # Lấy dữ liệu từ queue của Logic
            frame, summary = self.logic.frame_queue.get_nowait()
            self.display_cv2_image(frame)
            
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", summary)
        except queue.Empty:
            pass
        finally:
            self.root.after(20, self._update_camera_view)

    def on_close(self):
        self.logic.stop_camera_stream()
        self.root.destroy()