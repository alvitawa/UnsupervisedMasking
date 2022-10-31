import math
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt


class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

class ddict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def image_grid(imgs, rows, cols):
    """
    The ordering is row-first, i.e. the images [0,1,2,3] will be shown as
    0 1
    2 3

    :param imgs:
    :param rows:
    :param cols:
    :return:
    """
    if len(imgs) < rows * cols:
        imgs += list(Image.new('RGB', imgs[0].size, (256, 256, 256)) for _ in range(rows * cols - len(imgs)))
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def tensor_to_pil(image):
    return Image.fromarray(image.movedim(0, 2).cpu().numpy().astype(np.uint8))

def get_image_grid(images, rows=1, tensor=True):
    if tensor:
        pil_images = list(Image.fromarray(x.movedim(0, 2).cpu().numpy().astype(np.uint8)) for x in images)
    else:
        pil_images = images
    grid = image_grid(pil_images, rows, math.ceil(len(images) / rows))
    return grid

def save_image_grid(images, filename, location, rows=1, tensor=True):
    location = Path(location)
    filename = Path(filename + ".png")
    # pil_images = list(Image.fromarray(x.movedim(0, 2).cpu().numpy().astype(np.uint8)) for x in images)
    # grid = image_grid(pil_images, rows, math.ceil(len(images) / rows))
    grid = get_image_grid(images, rows, tensor)
    location.mkdir(parents=True, exist_ok=True)
    grid.save(Path(location, filename))

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_matrix(matrix):
    fig = plt.figure()
    plt.imshow(matrix, cmap='summer', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.xlabel(f'Axis 1 ({matrix.shape[1]})')
    plt.ylabel(f'Axis 0 ({matrix.shape[0]})')
    plot = fig2img(fig)
    return plot