import customtkinter as ctk
from gui.fruit_app import FruitDetectorGUI

if __name__ == "__main__":
    # Cài đặt DPI awareness cho Windows để hình ảnh sắc nét
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = ctk.CTk()
    app = FruitDetectorGUI(root)
    root.mainloop()