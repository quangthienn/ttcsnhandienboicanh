import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from ultralytics import YOLO

# Tải mô hình
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
yolo_model = YOLO("yolo11n.pt")

selected_files = []

# Xử lý ảnh
def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def translate_to_vietnamese(text):
    translator = Translator()
    result = translator.translate(text, src='en', dest='vi')
    return result.text

def process_image(file_path):
    img = Image.open(file_path)
    img_resized = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    status_label.config(text="⏳ Đang xử lý...")
    root.update()

    try:
        results = yolo_model(file_path)
        objects = results[0].names
        detected = list(set([objects[int(cls)] for cls in results[0].boxes.cls]))
        detected_objects = ", ".join(detected) if detected else "Không phát hiện đối tượng nào"

        caption_en = generate_caption(file_path)
        caption_vi = translate_to_vietnamese(caption_en)

        result_entry1.delete(0, tk.END)
        result_entry1.insert(0, detected_objects)

        result_entry2.delete(0, tk.END)
        result_entry2.insert(0, caption_en)

        result_entry3.delete(0, tk.END)
        result_entry3.insert(0, caption_vi)
    except Exception as e:
        result_entry1.delete(0, tk.END)
        result_entry1.insert(0, f"Lỗi: {e}")
    finally:
        status_label.config(text="✅ Hoàn tất")

def choose_single_image():
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        dropdown_menu.pack_forget()
        process_image(file_path)

def choose_multiple_images():
    global selected_files
    selected_files = list(filedialog.askopenfilenames(title="Chọn nhiều ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]))
    if not selected_files:
        return

    selected_var.set(selected_files[0])
    dropdown_menu['menu'].delete(0, 'end')
    for path in selected_files:
        display_name = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
        dropdown_menu['menu'].add_command(label=display_name, command=lambda p=path: on_dropdown_select(p))
    dropdown_menu.pack(pady=5)

def on_dropdown_select(file_path):
    selected_var.set(file_path)
    process_image(file_path)

# ========== GIAO DIỆN ==========
root = tk.Tk()
root.title("📷 Nhận diện ngữ cảnh bằng YOLOv8 + BLIP + Translate")
root.configure(bg="#1e1e1e")
root.geometry("800x600")

# Header cam
header = tk.Frame(root, bg="#ff5c00", height=40)
header.pack(fill="x")
tk.Label(header, text="📂 Font Select Image", bg="#ff5c00", fg="white", font=("Segoe UI", 12, "bold")).pack(side="left", padx=10)

# Nội dung chính
main_frame = tk.Frame(root, bg="#1e1e1e")
main_frame.pack(padx=20, pady=20)

# Cột trái: hình ảnh
image_label = tk.Label(main_frame, bg="#1e1e1e")
image_label.grid(row=0, column=0, rowspan=6, padx=20, pady=10)

# Cột phải: kết quả
tk.Label(main_frame, text="Các đối tượng phát hiện:", bg="#1e1e1e", fg="white").grid(row=0, column=1, sticky="w")
result_entry1 = tk.Entry(main_frame, width=40)
result_entry1.grid(row=1, column=1, pady=5)

tk.Label(main_frame, text="Mô tả (EN):", bg="#1e1e1e", fg="white").grid(row=2, column=1, sticky="w")
result_entry2 = tk.Entry(main_frame, width=40)
result_entry2.grid(row=3, column=1, pady=5)

tk.Label(main_frame, text="Mô tả (VI):", bg="#1e1e1e", fg="white").grid(row=4, column=1, sticky="w")
result_entry3 = tk.Entry(main_frame, width=40)
result_entry3.grid(row=5, column=1, pady=5)

# Nút chọn ảnh
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack()

tk.Button(btn_frame, text="🖼️ Chọn 1 ảnh", bg="#ff5c00", fg="white", font=("Segoe UI", 10, "bold"),
          width=15, height=2, command=choose_single_image).grid(row=0, column=0, padx=10, pady=10)

tk.Button(btn_frame, text="🖼️ Chọn nhiều ảnh", bg="#ff5c00", fg="white", font=("Segoe UI", 10, "bold"),
          width=15, height=2, command=choose_multiple_images).grid(row=0, column=1, padx=10, pady=10)

# Dropdown menu
selected_var = tk.StringVar(root)
dropdown_menu = tk.OptionMenu(root, selected_var, "")
dropdown_menu.configure(bg="#333333", fg="white")
dropdown_menu["menu"].configure(bg="#333333", fg="white")

# Trạng thái
status_label = tk.Label(root, text="", fg="lightgreen", bg="#1e1e1e")
status_label.pack()

root.mainloop()
