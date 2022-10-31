import os
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

domain_names = ['svhn', 'usps']
for domain in domain_names:
    new_path = f"./hw2_data_mock/digits/{domain}/val"
    os.makedirs(new_path, exist_ok=True)
    csv_name = f"./hw2_data/digits/{domain}/val.csv"
    # First, read data_list.
    with open(csv_name, 'r')as f:
        next(f)
        data_list = f.readlines()

    #print(data_list)
    img_paths = []

    # Then, read image from data_list.
    for data in data_list:
        img_paths.append(data[:-3]) # "00002.png",4<eol>
    #print(len(data_list))

    for img_name in tqdm(img_paths):
        path = os.path.join(f"./hw2_data/digits/{domain}/data", img_name)
        img = Image.open(path)
        img = img.save(os.path.join(new_path, img_name))

