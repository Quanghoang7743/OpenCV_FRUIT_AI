# Đề xuất cấu trúc thư mục cho dự án OpenCV Fruit AI

## 📁 Cấu trúc thư mục

```
OpenCV_FRUIT_AI/
│
├── README.md                          # Tài liệu hướng dẫn dự án
├── requirements.txt                   # Danh sách dependencies
├── .gitignore                         # Git ignore file
├── setup.py                           # Setup script (optional)
│
├── src/                               # Source code chính
│   ├── __init__.py
│   │
│   ├── gui/                           # GUI components
│   │   ├── __init__.py
│   │   └── fruit_app.py              # FruitDetectorGUI class
│   │
│   ├── core/                          # Core logic & business logic
│   │   ├── __init__.py
│   │   └── fruit_logic.py            # FruitDetectorLogic class
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── image_utils.py            # Image processing utilities
│       └── config.py                 # Configuration constants
│
├── models/                            # Model files
│   ├── .gitkeep                      # Keep folder in git
│   ├── fruit_detector_last.pt        # Trained model
│   └── fruit_detector_best.pt        # Best model (if available)
│
├── config/                            # Configuration files
│   ├── config.yaml                   # App configuration
│   └── model_config.yaml             # Model configuration
│
├── notebooks/                         # Jupyter notebooks
│   ├── fruit_recognition_training.ipynb
│   └── experiments/                  # Experimental notebooks
│
│
└── main.py                           # Entry point
```



### 1. requirements.txt
```
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
customtkinter>=5.0.0
numpy>=1.24.0
pyyaml>=6.0
```



