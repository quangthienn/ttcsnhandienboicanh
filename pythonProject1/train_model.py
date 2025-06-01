from ultralytics import YOLO

# 1. Load mô hình gốc YOLOv8n
model = YOLO('yolov8n.pt')  # Bạn có thể thay bằng yolov8s.pt nếu muốn mô hình lớn hơn

# 2. Huấn luyện mô hình bằng CPU
results = model.train(
    data='coco128.yaml',  # Đường dẫn file YAML chứa dữ liệu
    epochs=50,
    imgsz=640,
    batch=8,        # giảm batch size khi dùng CPU để tránh tràn RAM
    workers=0,      # dùng 1 luồng để tránh treo máy yếu
    device=0# ⚠️ CHẠY TRÊN CPU
)

# 3. Đánh giá mô hình sau huấn luyệnD:\pythonProject1\venv\Scripts\python.exe D:\pythonProject1\train_model.py
# Ultralytics 8.3.146  Python-3.10.9 torch-2.7.0+cpu
# Traceback (most recent call last):
#   File "D:\pythonProject1\train_model.py", line 7, in <module>
#     results = model.train(
#   File "D:\pythonProject1\venv\lib\site-packages\ultralytics\engine\model.py", line 791, in train
#     self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
#   File "D:\pythonProject1\venv\lib\site-packages\ultralytics\engine\trainer.py", line 121, in __init__
#     self.device = select_device(self.args.device, self.args.batch)
#   File "D:\pythonProject1\venv\lib\site-packages\ultralytics\utils\torch_utils.py", line 201, in select_device
#     raise ValueError(
# ValueError: Invalid CUDA 'device=0' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.
#
# torch.cuda.is_available(): False
# torch.cuda.device_count(): 0
# os.environ['CUDA_VISIBLE_DEVICES']: None
# See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.
metrics = model.val(device=0)

# 4. In hiệu năng
print("\n📊 Hiệu năng mô hình:")
print(f"mAP@0.5        : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95   : {metrics.box.map:.4f}")
print(f"Precision      : {metrics.box.precision:.4f}")
print(f"Recall         : {metrics.box.recall:.4f}")

# 5. Tính và in sai số (error)
error_map50 = 1 - metrics.box.map50
error_precision = 1 - metrics.box.precision

print("\n❌ Sai số:")
print(f"Sai số mAP@0.5       : {error_map50:.4f}")
print(f"Sai số Precision     : {error_precision:.4f}")
