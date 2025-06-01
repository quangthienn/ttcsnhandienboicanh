import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from ultralytics import YOLO
import torch

# Tải mô hình BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Tải mô hình YOLOv8
yolo_model = YOLO("yolo11n.pt")

selected_files = []  # Danh sách các ảnh đã chọn

# Hàm mô tả ảnh bằng BLIP
def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# Hàm dịch tiếng Anh sang tiếng Việt
def translate_to_vietnamese(text):
    translator = Translator()
    result = translator.translate(text, src='en', dest='vi')
    return result.text

# Hàm xử lý mô tả ảnh
def process_image(file_path):
    # Hiển thị ảnh
    img = Image.open(file_path)
    img_resized = img.resize((400, 300))
    img_tk = ImageTk.PhotoImage(img_resized)
    panel.configure(image=img_tk)
    panel.image = img_tk

    # Hiển thị trạng thái
    status_label.config(text="⏳ Đang xử lý...")
    root.update()

    try:
        # Nhận diện đối tượng bằng YOLO
        results = yolo_model(file_path)
        objects = results[0].names
        detected = list(set([objects[int(cls)] for cls in results[0].boxes.cls]))
        detected_objects = ", ".join(detected) if detected else "Không phát hiện đối tượng nào"

        # Mô tả bằng BLIP
        caption_en = generate_caption(file_path)

        # Dịch sang tiếng Việt
        caption_vi = translate_to_vietnamese(caption_en)

        # Hiển thị kết quả
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"🔍 Các đối tượng phát hiện: {detected_objects}\n")
        result_text.insert(tk.END, f"📝 Mô tả (EN): {caption_en}\n")
        result_text.insert(tk.END, f"📝 Mô tả (VI): {caption_vi}")
    except Exception as e:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"❌ Lỗi: {e}")
    finally:
        status_label.config(text="✅ Hoàn tất.")

# Chọn 1 ảnh và xử lý luôn
def choose_single_image():
    file_path = filedialog.askopenfilename(title="Chọn 1 ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        dropdown_menu.pack_forget()
        process_image(file_path)

# Chọn nhiều ảnh và hiển thị menu chọn
def choose_multiple_images():
    global selected_files
    selected_files = list(filedialog.askopenfilenames(title="Chọn nhiều ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]))
    if not selected_files:
        return

    # Hiện menu để chọn 1 ảnh để xử lý
    selected_var.set(selected_files[0])
    dropdown_menu['menu'].delete(0, 'end')
    for path in selected_files:
        display_name = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
        dropdown_menu['menu'].add_command(label=display_name, command=lambda p=path: on_dropdown_select(p))
    dropdown_menu.pack(pady=5)

# Khi người dùng chọn ảnh từ menu
def on_dropdown_select(file_path):
    selected_var.set(file_path)
    process_image(file_path)

# ======= GIAO DIỆN =======
root = tk.Tk()
root.title("📷 Nhận diện ngữ cảnh bằng YOLOv8 + BLIP + Translate")

panel = tk.Label(root)
panel.pack()

# Hai nút: Chọn 1 ảnh / nhiều ảnh
btn_single = tk.Button(root, text="🖼️ Chọn 1 ảnh", command=choose_single_image, height=2, width=20, bg='lightgreen')
btn_single.pack(pady=5)

btn_multi = tk.Button(root, text="🖼️ Chọn nhiều ảnh", command=choose_multiple_images, height=2, width=20, bg='orange')
btn_multi.pack(pady=5)

# Menu chọn ảnh nếu nhiều
selected_var = tk.StringVar(root)
dropdown_menu = tk.OptionMenu(root, selected_var, "")

status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
status_label.pack()

result_text = tk.Text(root, height=10, width=60, wrap=tk.WORD)
result_text.pack(padx=10, pady=10)

root.mainloop()
