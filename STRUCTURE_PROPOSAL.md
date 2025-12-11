# Đề xuất cấu trúc thư mục cho dự án OpenCV Fruit AI

## 📁 Cấu trúc thư mục được đề xuất

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
├── data/                              # Data directories
│   ├── input/                        # Input images for testing
│   ├── output/                       # Output images with detection
│   └── .gitkeep
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_fruit_logic.py
│   └── test_gui.py
│
├── docs/                              # Documentation
│   ├── API.md                        # API documentation
│   └── ARCHITECTURE.md               # Architecture documentation
│
└── main.py                           # Entry point
```

## 📝 Mô tả chi tiết các thư mục

### 1. **src/** - Source Code
- **gui/**: Chứa các component giao diện người dùng
  - `fruit_app.py`: Class GUI chính (FruitDetectorGUI)
  
- **core/**: Chứa business logic và xử lý chính
  - `fruit_logic.py`: Logic xử lý detection (FruitDetectorLogic)
  
- **utils/**: Các hàm tiện ích
  - `image_utils.py`: Các hàm xử lý ảnh chung
  - `config.py`: Các hằng số cấu hình

### 2. **models/** - Model Files
- Lưu trữ các file model đã train (.pt files)
- Nên thêm vào .gitignore nếu file quá lớn (>50MB)

### 3. **config/** - Configuration
- `config.yaml`: Cấu hình ứng dụng (paths, UI settings, etc.)
- `model_config.yaml`: Cấu hình model (classes, confidence threshold, etc.)

### 4. **notebooks/** - Jupyter Notebooks
- Chứa các notebook cho training và experimentation
- Folder `experiments/` cho các thử nghiệm khác

### 5. **data/** - Data Storage
- `input/`: Ảnh input để test
- `output/`: Ảnh output sau khi detect
- Có thể thêm vào .gitignore nếu không muốn commit data

### 6. **tests/** - Testing
- Unit tests cho các components
- Test coverage cho logic và GUI

### 7. **docs/** - Documentation
- API documentation
- Architecture documentation
- User guides

## 🔧 Các file cần tạo mới

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

### 2. .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Models (large files)
models/*.pt
*.pt
!models/.gitkeep

# Data (optional - uncomment if don't want to commit)
# data/input/*
# data/output/*
!data/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local
```

### 3. README.md
- Mô tả dự án
- Hướng dẫn cài đặt
- Hướng dẫn sử dụng
- Cấu trúc dự án

### 4. config/config.yaml
- Đường dẫn model mặc định
- Cấu hình camera
- Cấu hình UI
- Các tham số khác

## 🚀 Lợi ích của cấu trúc này

1. **Tổ chức rõ ràng**: Code được phân tách theo chức năng
2. **Dễ bảo trì**: Dễ dàng tìm và sửa code
3. **Mở rộng được**: Dễ thêm tính năng mới
4. **Chuyên nghiệp**: Tuân theo best practices của Python projects
5. **Tách biệt concerns**: GUI, Logic, Utils tách biệt rõ ràng
6. **Dễ test**: Tests được tổ chức riêng
7. **Version control**: .gitignore để quản lý file lớn

## 📋 Checklist khi di chuyển

- [ ] Tạo các thư mục mới
- [ ] Di chuyển file vào đúng vị trí
- [ ] Cập nhật import statements
- [ ] Tạo __init__.py files
- [ ] Tạo requirements.txt
- [ ] Tạo .gitignore
- [ ] Tạo config files
- [ ] Cập nhật README.md
- [ ] Test lại ứng dụng
- [ ] Update main.py imports

