from Module.builder import PREPROCESSES
import numpy as np
from PIL import Image
from tritonclient.utils import triton_to_np_dtype
import base64
from io import BytesIO
import numbers


@PREPROCESSES.register_module()
class Classifier_resize_trt():
    def __init__(self, resize_resolution=512, crop_resolution=448, dtype='FP32'):
        self.npdtype = triton_to_np_dtype(dtype)
        self.resize_resolution = resize_resolution
        self.crop_resolution = crop_resolution
        self.dtype = dtype

    def __call__(self, img, **kwargs):

        img = self.decodeBase64(img)
        origal_width, origal_height = img.size
        sample_img = img.convert('RGB')
        resize_img = self.resize(sample_img, self.resize_resolution)
        crop_img = self.center_crop(resize_img, self.crop_resolution)
        croped_img = np.array(crop_img)
        typed = croped_img.astype(self.npdtype)
        scaled = (typed - np.asarray((123.675, 116.28, 103.53), dtype=self.npdtype)) / np.asarray(
            (58.395, 57.12, 57.375), dtype=self.npdtype)
        ordered = np.transpose(scaled, (2, 0, 1))
        return ordered, 1.0, origal_width, origal_height

    def resize(self, img, size, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

    def center_crop(self, img, output_size):
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img.crop((j, i, j + tw, i + th))

    def decodeBase64(self, img):
        image = base64.b64decode(img)
        image = BytesIO(image)
        mat = Image.open(image)
        return mat
