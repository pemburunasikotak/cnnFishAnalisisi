import os
import shutil
import re

SOURCE_DIR = 'foto_ikan_real'
TARGET_DIR = 'dataset_real'

# 6 classes in foto_ikan_real
classes = [
    'Gerabah tidak segar',
    'Kembung tidak segar',
    'Kuniran tidak segar',
    'gerabah segar',
    'kembung segar',
    'kuniran segar'
]

# Ensure target directories exist
os.makedirs(os.path.join(TARGET_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, 'validation'), exist_ok=True)

def get_numeric_sort_key(filename):
    # Extract number from filename (e.g., '1.jpeg' -> 1)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 999999

for cls in classes:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    train_cls_dir = os.path.join(TARGET_DIR, 'train', cls)
    val_cls_dir = os.path.join(TARGET_DIR, 'validation', cls)
    
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)
    
    # List files in the source class directory
    files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # Sort files numerically
    files.sort(key=get_numeric_sort_key)
    
    print(f"Class: {cls} - Total found: {len(files)} files")
    
    # Take first 50 for train
    train_files = files[:50]
    # Take next 12 for validation
    val_files = files[50:62]
    
    print(f"  Copying {len(train_files)} to train...")
    for f in train_files:
        shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(train_cls_dir, f))
        
    print(f"  Copying {len(val_files)} to validation...")
    for f in val_files:
        shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(val_cls_dir, f))

print("Dataset splitting completed successfully!")
