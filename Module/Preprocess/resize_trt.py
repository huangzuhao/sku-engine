from Module.builder import PREPROCESSES
import numpy as np
from PIL import Image, ImageOps
from tritonclient.utils import triton_to_np_dtype
import base64
from io import BytesIO


@PREPROCESSES.register_module()
class Resize_trt():
    def __init__(self, width=1000, height=1000, dtype='FP32'):
        self.npdtype = triton_to_np_dtype(dtype)
        self.max_long_edge = max(width, height)
        self.max_short_edge = min(width, height)
        self.dtype = dtype

    def __call__(self, img, **kwargs):
        img = self.decodeBase64(img)

        if "direction" in kwargs:
            if kwargs["direction"] == 90:
                img = img.transpose(method=Image.ROTATE_90)
            elif kwargs["direction"] == -90:
                img = img.transpose(method=Image.ROTATE_270)

        origal_width, origal_height = img.size
        sample_img = img.convert('RGB')
        resized_img, scale_factor = self.imrescale(sample_img, Image.BILINEAR)
        pad_img, padding = self.get_pad_img(resized_img)
        resized = np.array(pad_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        # npdtype = triton_to_np_dtype(self.dtype)
        typed = resized.astype(self.npdtype)
        scaled = (typed - np.asarray((123.675, 116.28, 103.53), dtype=self.npdtype)) / np.asarray(
            (58.395, 57.12, 57.375),
            dtype=self.npdtype)
        ordered = np.transpose(scaled, (2, 0, 1))
        imgmeta = {}
        imgmeta["scale_factor"] = scale_factor
        imgmeta["padding"] = padding
        # return ordered, scale_factor, origal_width, origal_height
        return ordered, imgmeta, origal_width, origal_height

    def rescale_size(self, old_size):
        w, h = old_size
        scale_factor = min(self.max_long_edge / max(h, w), self.max_short_edge / min(h, w))
        new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
        return new_size, scale_factor

    def imrescale(self, img, interpolation='bilinear'):
        h, w = img.height, img.width
        new_size, scale_factor = self.rescale_size((w, h))
        rescaled_img = img.resize(new_size, interpolation)
        return rescaled_img, scale_factor

    def decodeBase64(self, img):
        image = base64.b64decode(img)
        image = BytesIO(image)
        mat = Image.open(image)
        return mat

    def get_pad_img(self, img, divsize=32):
        ori_w = img.width
        ori_h = img.height
        pad_w = int(np.ceil(ori_w / divsize)) * divsize - ori_w
        pad_h = int(np.ceil(ori_h / divsize)) * divsize - ori_h
        padding = (0, 0, pad_w, pad_h)
        img = ImageOps.expand(img, padding)
        return img, padding
