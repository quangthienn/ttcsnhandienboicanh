import os

label_root = r"D:\pythonProject1\datasetsmaill\labels"
image_root = r"D:\pythonProject1\datasetsmaill\images"

for split in ["train", "val"]:
    label_dir = os.path.join(label_root, split)
    image_dir = os.path.join(image_root, split)

    if not os.path.exists(label_dir):
        print(f"❌ Không tìm thấy thư mục: {label_dir}")
        continue

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, file)
        img_name = os.path.splitext(file)[0]

        # Tìm ảnh đi kèm (jpg hoặc png)
        image_path = None
        for ext in [".jpg", ".png"]:
            path = os.path.join(image_dir, img_name + ext)
            if os.path.exists(path):
                image_path = path
                break

        # Mặc định là giữ lại
        delete_image = False

        # Kiểm tra class_id trong file label
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                class_id = int(line.strip().split()[0])
                if class_id < 0 or class_id > 79:
                    delete_image = True
                    break

        if delete_image:
            os.remove(label_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            print(f"🗑️ Đã xóa ảnh và nhãn: {img_name}")
        else:
            print(f"✅ Giữ lại: {img_name}")
