import psutil
import os
from ultralytics import YOLO

process = psutil.Process(os.getpid())

def print_memory_stats():
    # Process usage (what this script is using)
    process_mem = process.memory_info().rss / 1024**2
    
    # System usage (what is left on the whole PC)
    system_mem = psutil.virtual_memory()
    available_mem = system_mem.available / 1024**2
    percent_used = system_mem.percent
    
    print(f"--- Memory Report ---")
    print(f"Process RAM Usage: {process_mem:.2f} MB")
    print(f"System Free RAM:    {available_mem:.2f} MB")
    print(f"Total System Load:  {percent_used}%")
    print("-" * 22)

model = YOLO(r"D:\Code\Python\brin\output\20260403_201314_767735\weights\best.pt")

print("Before inference:")
print_memory_stats()

results = model(r"D:\Code\Python\brin\surrounding-awareness-5-backup\valid\images\1775054153392_jpg.rf.b7474445b28fe5916cfb121d772a24a8.jpg")

print("After inference:")
print_memory_stats()