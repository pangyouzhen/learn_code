from pathlib import Path

from PIL import Image


def concat_img(path):
    img1 = Image.open(path / "0.jpg")
    im_list = []
    imgs = sorted(path.glob("*jpg"), key=lambda x: int(x.stem))
    for i in imgs:
        if i.stem == "0":
            continue
        im_list.append(Image.open(str(i)))

    pdf1_filename = path / "直线检测.pdf"

    img1.save(pdf1_filename, "PDF", resolution=100.0, save_all=True, append_images=im_list)


path = Path("./temp")
for i in path.iterdir():
    concat_img(i)
