import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

def convert_coco_json_to_yolo_txt(json_file_path, output_dir, image_dir):
    coco = COCO(json_file_path)
    os.makedirs(output_dir, exist_ok=True)

    image_id_to_filename = {img['id']: img['file_name'] for img in coco.dataset['images']}
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(categories)}

    for img_id in tqdm(coco.getImgIds(), desc="Converting"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]

        img_w = img_info['width']
        img_h = img_info['height']
        filename = os.path.splitext(img_info['file_name'])[0]
        label_file = os.path.join(output_dir, f"{filename}.txt")

        with open(label_file, "w") as f:
            for ann in anns:
                if ann.get("iscrowd", 0) == 1:
                    continue

                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w /= img_w
                h /= img_h

                category_id = ann['category_id']
                class_id = category_id_to_index[category_id]

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    return coco

def copy_images(image_ids, image_dir, save_dir, coco):
    os.makedirs(save_dir, exist_ok=True)
    for img_id in tqdm(image_ids, desc=f"Copying images to {save_dir}"):
        file_name = coco.loadImgs(img_id)[0]['file_name']
        src_path = os.path.join(image_dir, file_name)
        dst_path = os.path.join(save_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    # === Đường dẫn thư mục dữ liệu COCO ===
    root = "datasetscoco"
    annotations = os.path.join(root, "annotations")
    images = os.path.join(root, "images")

    # === Train set ===
    train_json = os.path.join(annotations, "instances_train2017.json")
    train_img_dir = os.path.join(images, "train2017")
    train_label_output = os.path.join("datasets", "labels", "train")
    train_image_output = os.path.join("datasets", "images", "train")

    train_coco = convert_coco_json_to_yolo_txt(train_json, train_label_output, train_img_dir)
    copy_images(train_coco.getImgIds(), train_img_dir, train_image_output, train_coco)

    # === Validation set ===
    val_json = os.path.join(annotations, "instances_val2017.json")
    val_img_dir = os.path.join(images, "val2017")
    val_label_output = os.path.join("datasets", "labels", "val")
    val_image_output = os.path.join("datasets", "images", "val")

    val_coco = convert_coco_json_to_yolo_txt(val_json, val_label_output, val_img_dir)
    copy_images(val_coco.getImgIds(), val_img_dir, val_image_output, val_coco)

    print("✅ Hoàn tất chuyển đổi COCO -> YOLO và sao chép ảnh!")
