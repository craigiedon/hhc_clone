import os

import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import copy

import chainer
from chainer import Variable

IMAGE_NET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def data_process(batch, converter=chainer.dataset.concat_examples, device=None, volatile=False):
    if volatile:
        with chainer.no_backprop_mode():
            return Variable(converter(batch, device))
    else:
        return Variable(converter(batch, device))

def output2img_single(y):
    y = chainer.cuda.to_cpu(y.data)
    y = np.squeeze(y).transpose((1, 2, 0))
    return y * 255.

def output2img_cv2(y):
    y = chainer.cuda.to_cpu(y.data)
    return np.asarray(y.transpose((0, 2, 3, 1)) * 255)

def append_horizontal_images(images):
    widths, heights = zip(*([i.shape[0], i.shape[1]] for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        im = Image.fromarray(np.uint8(im))
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    return new_im

def append_vertical_images(images):
    if isinstance(images[0], np.ndarray):
        widths, heights = zip(*([i.shape[0], i.shape[1]] for i in images))
    else:
        widths, heights = zip(*([i.size[0], i.size[1]] for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        im = Image.fromarray(np.uint8(im))
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    
    return new_im


def display_image(f, dataset, dst, device, n=1):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_dir = '{}/preview'.format(dst)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        all_img = []
        for i in range(n):
            idx = np.random.randint(0, len(dataset))

            # print(dataset[idx])
            A, _ = dataset[idx]
            A = copy.deepcopy(A)
            name = str(idx)

            with chainer.using_config('train', False):
                decoded = f(data_process([A], device=device, volatile=True))
                decoded = output2img_single(decoded)
                # print('output2img_single', np.amax(decoded), np.amin(decoded))
                # decoded += IMAGE_NET_MEAN # NO need to add the mean value here
                # print('+= IMAGE_NET_MEAN', np.amax(decoded), np.amin(decoded))
                decoded = decoded.astype(np.int8)
                # print('np.int8', np.amax(decoded), np.amin(decoded))

            name = os.path.splitext(name)[0]
            preview_path = preview_dir + '/{}_epoch__{}_iter__{}.png'.format(trainer.updater.epoch, trainer.updater.iteration, name)
            A_orig = A.transpose((1, 2, 0))*255.
            # A_orig += IMAGE_NET_MEAN

            merged = append_horizontal_images([A_orig, decoded])
            all_img.append(copy.deepcopy(merged))

        merged = append_vertical_images(all_img)
        merged.save(preview_path)

    return make_image
