from pathlib import Path

from PIL import Image

img1 = Image.open("./temp/0.jpg")
im_list = []
path = Path("./temp")
imgs = sorted(path.glob("*jpg"), key=lambda x: int(x.stem))
for i in imgs:
    if i.stem == "0":
        continue
    im_list.append(Image.open(str(i)))

pdf1_filename = "./face.pdf"

img1.save(pdf1_filename, "PDF", resolution=100.0, save_all=True, append_images=im_list)
