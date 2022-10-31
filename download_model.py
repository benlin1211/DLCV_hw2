import gdown

url = "https://drive.google.com/file/d/1ti8uTNvWn6C2yuv8RneWXxhocHlB8eEu/view?usp=sharing"
output = "ckpt2-2.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1FKYJ5LfgZv_pStE8XQ4Ocja7WQz4D6YW/view?usp=sharing"
output = "ckpt2-3_svhn.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1sNDHIshVGH335ANtUCFp7J6rbvPzZhm-/view?usp=sharing"
output = "ckpt2-3_usps.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

