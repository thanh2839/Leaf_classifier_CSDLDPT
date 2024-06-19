import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import classify_leaf_little as classifier
# Hàm để chọn và hiển thị ảnh chính
def load_main_image():
    global main_image_label, main_image_path
    main_image_path = filedialog.askopenfilename()
    if main_image_path:
        img = Image.open(main_image_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        main_image_label.config(image=img)
        main_image_label.image = img
    classifier.feature_extract(main_image_label)
    print(classifier.feature_extract)

# Hàm để hiển thị thêm ba ảnh
def show_similar_images():
    similar_images_paths = [main_image_path] * 3  # Sử dụng ảnh chính làm ví dụ
    for i in range(3):
        img = Image.open(similar_images_paths[i])
        img.thumbnail((100, 100))
        img = ImageTk.PhotoImage(img)
        similar_image_labels[i].config(image=img)
        similar_image_labels[i].image = img

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Hiển thị hình ảnh")
root.geometry('500x500')  # Đặt kích thước cửa sổ thành 500x500

# Nút để tải ảnh chính
load_button = tk.Button(root, text="Tải ảnh chính", command=load_main_image)
load_button.pack(pady=10)

# Khung hiển thị ảnh chính
main_image_label = tk.Label(root)
main_image_label.pack(pady=10)

# Nút sự kiện
event_button = tk.Button(root, text="Hiển thị ảnh tương tự", command=show_similar_images)
event_button.pack(pady=10)

# Khung hiển thị ba ảnh tương tự
similar_image_frame = tk.Frame(root)
similar_image_frame.pack(pady=10)
similar_image_labels = [tk.Label(similar_image_frame) for _ in range(3)]
for label in similar_image_labels:
    label.pack(side=tk.LEFT, padx=5)

# Chạy vòng lặp chính
root.mainloop()
