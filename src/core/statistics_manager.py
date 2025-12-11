"""
Statistics Manager - Quản lý và lưu trữ dữ liệu thống kê detection
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional


class StatisticsManager:
    """Quản lý thống kê detections"""
    
    def __init__(self):
        self.detection_history = []
        self.class_counts = defaultdict(int)
        self.confidence_scores = defaultdict(list)
        self.session_count = 0
        self.total_detections = 0
        
    def add_detection(self, detections, model_names: Optional[Dict] = None, timestamp=None):
        """
        Thêm detection vào lịch sử
        
        Args:
            detections: Detection boxes từ YOLO
            model_names: Dictionary mapping class_id to class_name
            timestamp: Timestamp của detection (mặc định là now)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.session_count += 1
        
        detection_data = {
            'timestamp': timestamp,
            'session_id': self.session_count,
            'detections': [],
            'total': len(detections) if detections else 0
        }
        
        session_class_counts = defaultdict(int)
        
        if detections:
            for det in detections:
                cls = int(det.cls.item()) if hasattr(det.cls, 'item') else int(det.cls)
                conf = float(det.conf.item()) if hasattr(det.conf, 'item') else float(det.conf)
                
                # Lấy class name
                if model_names:
                    class_name = model_names.get(cls, f"Class_{cls}")
                else:
                    class_name = f"Class_{cls}"
                
                detection_data['detections'].append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf
                })
                
                # Cập nhật thống kê tổng hợp
                self.class_counts[class_name] += 1
                self.confidence_scores[class_name].append(conf)
                session_class_counts[class_name] += 1
                self.total_detections += 1
        
        detection_data['class_counts'] = dict(session_class_counts)
        self.detection_history.append(detection_data)
        
        return detection_data
    
    def get_summary_stats(self) -> Dict:
        """Lấy thống kê tổng hợp"""
        avg_confidences = {
            cls: sum(scores) / len(scores) if scores else 0.0
            for cls, scores in self.confidence_scores.items()
        }
        
        max_confidences = {
            cls: max(scores) if scores else 0.0
            for cls, scores in self.confidence_scores.items()
        }
        
        min_confidences = {
            cls: min(scores) if scores else 0.0
            for cls, scores in self.confidence_scores.items()
        }
        
        # Tính confidence trung bình tổng thể
        all_confidences = []
        for scores in self.confidence_scores.values():
            all_confidences.extend(scores)
        
        overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            'total_detections': self.total_detections,
            'total_sessions': self.session_count,
            'class_counts': dict(self.class_counts),
            'average_confidences': avg_confidences,
            'max_confidences': max_confidences,
            'min_confidences': min_confidences,
            'overall_avg_confidence': overall_avg_confidence,
            'num_classes_detected': len(self.class_counts)
        }
    
    def get_class_distribution(self) -> Dict[str, float]:
        """Lấy tỷ lệ phần trăm của từng class"""
        if self.total_detections == 0:
            return {}
        
        distribution = {}
        for class_name, count in self.class_counts.items():
            distribution[class_name] = (count / self.total_detections) * 100
        
        return distribution
    
    def get_recent_detections(self, n=10) -> List[Dict]:
        """Lấy n detections gần nhất"""
        return self.detection_history[-n:] if len(self.detection_history) > n else self.detection_history
    
    def reset_statistics(self):
        """Reset tất cả thống kê"""
        self.detection_history = []
        self.class_counts = defaultdict(int)
        self.confidence_scores = defaultdict(list)
        self.session_count = 0
        self.total_detections = 0
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID (backup method)"""
        class_names = {
            0: 'apple', 1: 'banana', 2: 'cherry', 3: 'lemon',
            4: 'orange', 5: 'peach', 6: 'pear', 
            7: 'strawberry', 8: 'watermelon'
        }
        return class_names.get(class_id, f'class_{class_id}')

