import gdown

url = "https://drive.google.com/file/d/1ti8uTNvWn6C2yuv8RneWXxhocHlB8eEu/view?usp=sharing"
output = "ckpt2-2.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

