import os
import shutil
import random

# Đường dẫn gốc
image_src = "datasets/images/train"
label_src = "datasets/labels/train"

# Đường dẫn đích
base_dst = "datasetsmaill"
image_dst_train = os.path.join(base_dst, "images/train")
image_dst_val = os.path.join(base_dst, "images/val")
label_dst_train = os.path.join(base_dst, "labels/train")
label_dst_val = os.path.join(base_dst, "labels/val")

# Tạo thư mục đích nếu chưa có
for folder in [image_dst_train, image_dst_val, label_dst_train, label_dst_val]:
    os.makedirs(folder, exist_ok=True)

# Lấy danh sách file ảnh (chỉ lấy các ảnh có nhãn)
image_files = [f for f in os.listdir(image_src) if f.endswith(('.jpg', '.png'))]
image_files = [f for f in image_files if os.path.exists(os.path.join(label_src, f.replace('.jpg', '.txt').replace('.png', '.txt')))]

# Lấy ngẫu nhiên 1/20 dữ liệu
sample_size = max(1, len(image_files) // 20)
sample_files = random.sample(image_files, sample_size)

# Shuffle rồi chia 80/20
random.shuffle(sample_files)
split_idx = int(len(sample_files) * 0.8)
train_files = sample_files[:split_idx]
val_files = sample_files[split_idx:]

def copy_data(files, image_dst, label_dst):
    for img_file in files:
        label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        shutil.copy(os.path.join(image_src, img_file), os.path.join(image_dst, img_file))
        shutil.copy(os.path.join(label_src, label_file), os.path.join(label_dst, label_file))

# Copy dữ liệu
copy_data(train_files, image_dst_train, label_dst_train)
copy_data(val_files, image_dst_val, label_dst_val)

print(f"✅ Tạo thành công dataset nhỏ ({sample_size} ảnh): {len(train_files)} train / {len(val_files)} val")
