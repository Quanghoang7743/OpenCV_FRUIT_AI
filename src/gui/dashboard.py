"""
Dashboard Component - Hiển thị thống kê và charts
"""
import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Optional


class StatisticsDashboard(ctk.CTkFrame):
    """Dashboard hiển thị thống kê với charts"""
    
    def __init__(self, parent, statistics_manager):
        super().__init__(parent)
        self.stats_manager = statistics_manager
        self.current_figures = []  # Store figure references
        self.setup_ui()
    
    def setup_ui(self):
        """Thiết lập UI cho dashboard"""
        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.summary_tab = self.tabview.add("Summary")
        self.distribution_tab = self.tabview.add("Distribution")
        self.counts_tab = self.tabview.add("Counts")
        self.confidence_tab = self.tabview.add("Confidence")
        
        # Setup each tab
        self.create_summary_view()
        self.create_distribution_chart()
        self.create_counts_chart()
        self.create_confidence_chart()
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            self, 
            text="Tại lại Dashboard", 
            command=self.refresh_all,
            width=200
        )
        refresh_btn.pack(pady=10)
    
    def create_summary_view(self):
        """Tạo view tổng hợp thống kê"""
        summary = self.stats_manager.get_summary_stats()
        
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self.summary_tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            scroll_frame, 
            text="SUMMARY STATISTICS", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        
        # Overall stats
        stats_frame = ctk.CTkFrame(scroll_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        total_detections_label = ctk.CTkLabel(
            stats_frame,
            text=f"Total Detections: {summary['total_detections']}",
            font=ctk.CTkFont(size=16)
        )
        total_detections_label.pack(pady=10)
        
        total_sessions_label = ctk.CTkLabel(
            stats_frame,
            text=f"Total Sessions: {summary['total_sessions']}",
            font=ctk.CTkFont(size=16)
        )
        total_sessions_label.pack(pady=5)
        
        avg_conf_label = ctk.CTkLabel(
            stats_frame,
            text=f"Overall Avg Confidence: {summary['overall_avg_confidence']:.2%}",
            font=ctk.CTkFont(size=16)
        )
        avg_conf_label.pack(pady=5)
        
        classes_label = ctk.CTkLabel(
            stats_frame,
            text=f"Classes Detected: {summary['num_classes_detected']}",
            font=ctk.CTkFont(size=16)
        )
        classes_label.pack(pady=5)
        
        # Class details
        if summary['class_counts']:
            separator = ctk.CTkLabel(
                scroll_frame,
                text="─" * 50,
                text_color="gray"
            )
            separator.pack(pady=20)
            
            class_title = ctk.CTkLabel(
                scroll_frame,
                text="CLASS DETAILS",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            class_title.pack(pady=10)
            
            for class_name in sorted(summary['class_counts'].keys()):
                class_frame = ctk.CTkFrame(scroll_frame)
                class_frame.pack(fill="x", padx=10, pady=5)
                
                count = summary['class_counts'][class_name]
                avg_conf = summary['average_confidences'].get(class_name, 0)
                max_conf = summary['max_confidences'].get(class_name, 0)
                min_conf = summary['min_confidences'].get(class_name, 0)
                
                class_label = ctk.CTkLabel(
                    class_frame,
                    text=f"🍎 {class_name.capitalize()}: {count} detections",
                    font=ctk.CTkFont(size=14, weight="bold")
                )
                class_label.pack(anchor="w", padx=10, pady=5)
                
                conf_label = ctk.CTkLabel(
                    class_frame,
                    text=f"   Avg: {avg_conf:.2%} | Max: {max_conf:.2%} | Min: {min_conf:.2%}",
                    font=ctk.CTkFont(size=12),
                    text_color="gray"
                )
                conf_label.pack(anchor="w", padx=20, pady=2)
        else:
            no_data_label = ctk.CTkLabel(
                scroll_frame,
                text="No detection data available.\nProcess some images to see statistics!",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            no_data_label.pack(pady=50)
    
    def create_distribution_chart(self):
        """Tạo pie chart phân bố"""
        distribution = self.stats_manager.get_class_distribution()
        
        if not distribution:
            no_data_label = ctk.CTkLabel(
                self.distribution_tab,
                text="No data available",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(expand=True)
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#212121')  # Dark background
        ax.set_facecolor('#2b2b2b')
        
        # Set color scheme for dark mode
        colors = plt.cm.Set3(np.linspace(0, 1, len(distribution)))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            distribution.values(),
            labels=distribution.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'color': 'white', 'fontsize': 10}
        )
        
        ax.set_title('Fruit Distribution', color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.distribution_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        self.current_figures.append(fig)
    
    def create_counts_chart(self):
        """Tạo bar chart số lượng"""
        summary = self.stats_manager.get_summary_stats()
        class_counts = summary['class_counts']
        
        if not class_counts:
            no_data_label = ctk.CTkLabel(
                self.counts_tab,
                text="No data available",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(expand=True)
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#212121')
        ax.set_facecolor('#2b2b2b')
        
        # Prepare data
        classes = sorted(class_counts.keys())
        counts = [class_counts[cls] for cls in classes]
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
        bars = ax.bar(classes, counts, color=colors)
        
        # Customize
        ax.set_xlabel('Fruit Type', color='white', fontsize=12)
        ax.set_ylabel('Count', color='white', fontsize=12)
        ax.set_title('Detection Counts by Fruit Type', color='white', fontsize=14, fontweight='bold', pad=20)
        ax.tick_params(colors='white')
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.counts_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        self.current_figures.append(fig)
    
    def create_confidence_chart(self):
        """Tạo bar chart confidence scores"""
        summary = self.stats_manager.get_summary_stats()
        avg_confidences = summary['average_confidences']
        
        if not avg_confidences:
            no_data_label = ctk.CTkLabel(
                self.confidence_tab,
                text="No data available",
                font=ctk.CTkFont(size=16)
            )
            no_data_label.pack(expand=True)
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#212121')
        ax.set_facecolor('#2b2b2b')
        
        # Prepare data
        classes = sorted(avg_confidences.keys())
        avg_confs = [avg_confidences[cls] * 100 for cls in classes]  # Convert to percentage
        max_confs = [summary['max_confidences'].get(cls, 0) * 100 for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width/2, avg_confs, width, label='Average', color='#1f77b4')
        bars2 = ax.bar(x + width/2, max_confs, width, label='Maximum', color='#ff7f0e')
        
        # Customize
        ax.set_xlabel('Fruit Type', color='white', fontsize=12)
        ax.set_ylabel('Confidence (%)', color='white', fontsize=12)
        ax.set_title('Average and Maximum Confidence by Fruit Type', 
                    color='white', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax.set_ylim([0, 105])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', color='white', fontsize=9)
        
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.confidence_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        self.current_figures.append(fig)
    
    def refresh_all(self):
        """Refresh tất cả charts và views"""
        # Clear existing figures
        for fig in self.current_figures:
            plt.close(fig)
        self.current_figures = []
        
        # Clear tabs
        for tab_name in ["Summary", "Distribution", "Counts", "Confidence"]:
            if tab_name in self.tabview._tab_dict:
                # Clear tab content
                for widget in self.tabview._tab_dict[tab_name].winfo_children():
                    widget.destroy()
        
        # Recreate views
        self.summary_tab = self.tabview.tab("Summary")
        self.distribution_tab = self.tabview.tab("Distribution")
        self.counts_tab = self.tabview.tab("Counts")
        self.confidence_tab = self.tabview.tab("Confidence")
        
        self.create_summary_view()
        self.create_distribution_chart()
        self.create_counts_chart()
        self.create_confidence_chart()

