from rembg import remove
from PIL import Image
import os

# Đường dẫn tới ảnh đầu vào và đầu ra
def count_files(directory):
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.isdir(directory):
        print(f"{directory} không phải là một thư mục.")
        return -1
    
    # Lấy danh sách các tệp tin trong thư mục
    files = os.listdir(directory)
    
    # Sử dụng len() để đếm số lượng tệp tin
    file_count = len(files)
    
    return file_count

folder_path = './RGB-leaf'
folder_leaf = '3. Populus nigra'
# Gọi hàm đếm số lượng tệp tin
# file_count = count_files(f"{folder_path}/{folder_leaf}")
# print (file_count)
folder = sorted(os.listdir('./RGB-leaf'))
print(len(folder))
count = 0 
for i in range (0, len(folder)):
    folder_number  = folder[i].split(".")[0].strip()
    #print(folder_number)
    if int(folder_number) < 10:
        folder_number = f'0{folder_number}'
    else:
        folder_number = str(folder_number)
    
    folder_leaf = folder[i]
    file_count = count_files(f"{folder_path}/{folder_leaf}")
    for j in range(1, file_count + 1):
        if j < 10:
            file_number = f'0{j}'
        else:
            file_number = str(j)
        inp = f"{folder_path}/{folder_leaf}/iPAD2_C{folder_number}_EX{file_number}.JPG"
        print(inp)
        out = f"./remove/{folder_leaf}/iPAD2_C{folder_number}_EX{file_number}.JPG"
        print(f"remove : {out}")
        #tách nền

        input_image = Image.open(inp)
        output_image = remove(input_image)
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')
        output_image.save(out)

# for i in range(1, file_count + 1):
#     if i < 10:
#         file_number = f'0{i}'
#     else:
#         file_number = str(i)
#     inp = f"{folder_path}/{folder_leaf}/iPAD2_C03_EX{file_number}.JPG"
#     print(inp)
#     out = f"./remove/{folder_leaf}/iPAD2_C03_EX{file_number}.JPG"
#     print(out)

#     input_image = Image.open(inp)
#     output_image = remove(input_image)
    
#     if output_image.mode == 'RGBA':
#         output_image = output_image.convert('RGB')
#     output_image.save(out)
    #inp = folder_path + 'iPAD2_C01_EX{i}.JPG'
# inp = './RGB-leaf/1. Quercus suber/iPAD2_C01_EX01.JPG'
# inp = './RGB-leaf/1. Quercus suber/iPAD2_C01_EX02.JPG'
# inp = './RGB-leaf/1. Quercus suber/iPAD2_C01_EX03.JPG'

# out = './remove/iPAD2_C01_EX01.JPG_removeBG.JPG'


# # Mở ảnh đầu vào
# input_image = Image.open(inp)

# # Tách nền khỏi ảnh
# output_image = remove(input_image)

# # Chuyển đổi ảnh kết quả sang chế độ RGB (nếu nó đang ở chế độ RGBA)
# if output_image.mode == 'RGBA':
#     output_image = output_image.convert('RGB')

# # Lưu ảnh kết quả
# output_image.save(out)
