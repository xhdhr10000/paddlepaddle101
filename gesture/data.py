import os
import numpy as np
from PIL import Image, ImageEnhance

DATA_DIM = 224

train_list = os.path.join('dataset', 'train_list.txt')
test_list = os.path.join('dataset', 'val_list.txt')

def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.LANCZOS)
    return img

def rotate_image(img):
    angle = np.random.randint(-10, 11)
    img = img.rotate(angle)
    return img

def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img

def load_image(path, rotate=True, color_jitter=True):
    im = Image.open(path)

    if rotate: im = rotate_image(im)
#    im = random_crop(im, DATA_DIM)
    if color_jitter:
        im = distort_color(im)
    if np.random.randint(0, 2) == 1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

#    im = im.resize((64, 64), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 3, 64, 64).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im

def load_dataset(path):
    s = []
    with open(path) as f:
        lines = f.read().splitlines()
        for line in lines:
            path = line.split(' ')[0]
            path = os.path.join('dataset', path)
            label = line.split(' ')[1]
            for i in range(10):
                s.append((path, label))
    np.random.shuffle(s)
    return s

train_set = load_dataset(train_list)
test_set = load_dataset(test_list)

def read_data(train_or_test):
    if train_or_test == 'train':
        def reader():
            for d in train_set:
                img = load_image(d[0])
                yield (img, d[1])
    else:
        def reader():
            for d in test_set:
                img = load_image(d[0])
                yield (img, d[1])
    return reader
