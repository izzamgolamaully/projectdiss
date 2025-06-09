import os
import random
import shutil

# Paths to images and labels
image_dir = "dataset/images"
label_dir = "dataset/labels"

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)  # Shuffle to ensure randomness

# Compute split sizes
train_size = int(len(image_files) * train_ratio)
val_size = int(len(image_files) * val_ratio)

# Split data
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

# Function to move files
def move_files(file_list, split):
    for file in file_list:
        # Move image
        shutil.move(os.path.join(image_dir, file), os.path.join(f"dataset/images/{split}", file))
        # Move corresponding label file
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(label_dir, label_file)):
            shutil.move(os.path.join(label_dir, label_file), os.path.join(f"dataset/labels/{split}", label_file))

# Move files to respective folders
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print("Dataset successfully split!")
