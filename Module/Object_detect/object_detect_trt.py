import numpy

from Module.builder import DETECTS
from functools import partial
import queue
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
# import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
import torch
import torch.nn.functional as F
import cv2
import pycocotools.mask as maskUtils


@DETECTS.register_module()
class Object_detect_trt():
    def __init__(self, modelname=None, threshold=0.6, with_mask=False, result_type='sku', model_version='',
                 sku_list=None):
        self.model_name_ = modelname
        self.thresolds = threshold
        self.with_mask = with_mask
        self.result_type = result_type
        self.sku_list = sku_list
        self.model_version = model_version
        self.triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False)
        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=modelname, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))

        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=modelname, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))

        self.max_batch_size, self.input_name, self.output_name, self.dtype = parse_model_grpc(self.model_metadata,
                                                                                              self.model_config.config)

    def forward(self, img, imgmeta):
        repeatdata = []
        repeatdata.append(img)
        batched_image_data = np.stack(repeatdata, axis=0)
        responses = []
        sent_count = 0
        user_data = UserData()
        try:
            for inputs, outputs in requestGenerator(batched_image_data, self.input_name, self.output_name, self.dtype):
                sent_count += 1
                self.triton_client.async_infer(self.model_name_, inputs, partial(completion_callback, user_data),
                                               request_id=str(sent_count), model_version=self.model_version,
                                               outputs=outputs)
        except InferenceServerException as e:
            print("inference failed: " + str(e))

        processed_count = 0
        while processed_count < sent_count:
            (results, error) = user_data._completed_requests.get()
            processed_count += 1
            if error is not None:
                print("inference failed: " + str(error))
            responses.append(results)
        list_results = []
        for response in responses:
            if self.with_mask:
                list_results.append(self.gather_mask_result(response, self.output_name, imgmeta, img.shape))
            else:
                list_results.append(self.gather_bbox_result(response, self.output_name, imgmeta, img.shape))
        return list_results[0]

    def gather_bbox_result(self, results, output_name, imgmeta, img_shape):
        pad_w = imgmeta["padding"][2]
        pad_h = imgmeta["padding"][3]
        imgWH = numpy.array([img_shape[2] - pad_w, img_shape[1] - pad_h, img_shape[2] - pad_w, img_shape[1] - pad_h])
        output_array = []
        final_result = []
        for i, name in enumerate(output_name):
            output_array.append(results.as_numpy(name)[0])
        num_data = int(output_array[0][0])
        # bbox_data = (output_array[1] / scale_factor).astype(int).tolist()
        bbox_data = ((output_array[1]) / imgWH).tolist()
        score_data = output_array[2].tolist()
        class_data = output_array[3].astype(int).tolist()

        if num_data > 0:
            for j in range(num_data):
                bbox_data[j].extend([round(score_data[j], 3), self.sku_list[class_data[j]]])
                final_result.append(bbox_data[j])

        # print(final_result)

        return self.filter_sku_with_threshold(final_result)

    def gather_mask_result(self, results, output_name, imgmeta, img_shape):
        pad_w = imgmeta["padding"][2]
        pad_h = imgmeta["padding"][3]
        imgWH = numpy.array([img_shape[2] - pad_w, img_shape[1] - pad_h, img_shape[2] - pad_w, img_shape[1] - pad_h])
        output_array = []
        final_result = []
        for i, name in enumerate(output_name):
            output_array.append(results.as_numpy(name)[0])
        num_data = int(output_array[0][0])
        # bbox_data = (output_array[1] / imgmeta).astype(int).tolist()
        bbox_data = ((output_array[1]) / imgWH).tolist()
        score_data = output_array[2].tolist()
        class_data = output_array[3].astype(int).tolist()

        if num_data > 0:
            # segm_result = self.get_seg_masks(torch.from_numpy(output_array[4]),torch.from_numpy(output_array[3]),self.sku_list,img_shape,num_data)
            segm_result = get_seg_masks(torch.from_numpy(output_array[4]), torch.from_numpy(output_array[1]), img_shape,
                                        num_data)
            encoded_mask_results = encode_mask_results(segm_result)
            mask_data = get_point(encoded_mask_results)

            for j in range(num_data):
                bbox_data[j].extend([round(score_data[j], 3), self.sku_list[class_data[j]],
                                     (mask_data[j] / imgmeta["scale_factor"]).astype(int).tolist()])
                final_result.append(bbox_data[j])
                # print(bbox_data[j])
        # print(final_result)

        return self.filter_sku_with_threshold(final_result)

    def filter_sku_with_threshold(self, all_result):
        # "sku": [[100, 200, 300, 400, 0.93, "sku1"],
        # #       [100, 200, 300, 400, 0.93, "sku2"],
        # #       [100, 200, 300, 400, 0.93, "sku3"]]
        # sku_threshold = {"Default": 0.45, "Specify": {"ty1": 0.7,"ty1": 0.8}}
        filtered_result = []
        for sku in all_result:
            if "Specify" in self.thresolds and sku[5] in self.thresolds["Specify"]:
                if sku[4] > self.thresolds["Specify"][sku[5]]:
                    filtered_result.append(sku)
            else:
                if sku[4] > self.thresolds["Default"]:
                    filtered_result.append(sku)
        return filtered_result


def get_seg_masks(mask_pred, det_bboxes, img_shape, nums):
    """Get segmentation masks from mask_pred and bboxes.

    Args:
        mask_pred (Tensor or ndarray): shape (n, #class, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multi-scale testing,
            it will be converted to numpy array outside of this method.
        det_bboxes (Tensor): shape (n, 4/5)
        det_labels (Tensor): shape (n, )
        img_shape (Tensor): shape (3, )
        rcnn_test_cfg (dict): rcnn testing config
        ori_shape: original image size

    Returns:
        list[list]: encoded masks
    """
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.sigmoid()
    else:
        mask_pred = det_bboxes.new_tensor(mask_pred)

    cls_segms = []
    bboxes = det_bboxes[:, :4]
    # labels = det_labels
    img_h, img_w = img_shape[1:]

    N = nums
    num_chunks = N

    chunks = torch.chunk(torch.arange(N), num_chunks)

    threshold = 0.6
    im_mask = torch.zeros(
        N,
        img_h,
        img_w,
        dtype=torch.bool)

    # mask_pred = mask_pred[range(N), labels][:, None]
    # if not self.class_agnostic:
    #     mask_pred = mask_pred[range(N), labels][:, None]

    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            mask_pred[inds],
            bboxes[inds],
            img_h,
            img_w,
            skip_empty=True)

        masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)

        im_mask[(inds,) + spatial_inds] = masks_chunk

    for i in range(N):
        cls_segms.append(im_mask[i].cpu().numpy())
    return cls_segms


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.

    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]
    C, H, W = masks.shape
    masks = masks.view(-1, 1, H, W)

    img_y = torch.arange(y0_int, y1_int, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    if isinstance(mask_results, tuple):  # mask scoring
        cls_segms, cls_mask_scores = mask_results
    else:
        cls_segms = mask_results

    encoded_mask_results = []
    for cls_segm in cls_segms:
        encoded_mask_results.append(
            maskUtils.encode(np.array(cls_segm[:, :, np.newaxis], order='F', dtype='uint8'))[0])  # encoded with RLE
    if isinstance(mask_results, tuple):
        return encoded_mask_results, cls_mask_scores
    else:
        return encoded_mask_results


def get_point(segms):
    points = []
    for i in range(len(segms)):
        mask = maskUtils.decode(segms[i])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        points_contour = []
        for inx in range(len(contours)):
            points_contour.append(len(contours[inx]))

        index_contour = points_contour.index(max(points_contour))
        epsilon = 0.0004 * cv2.arcLength(contours[index_contour], True)
        approx = cv2.approxPolyDP(contours[index_contour], epsilon, True)
        approx = np.squeeze(approx, axis=1)
        points.append(approx)

    return points


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model_grpc(model_metadata, model_config):
    input_metadata = model_metadata.inputs[0]
    output_metadataname = []
    for name in model_metadata.outputs:
        output_metadataname.append(name.name)
    max_batch_size = 1
    return (max_batch_size, input_metadata.name, output_metadataname, input_metadata.datatype)


def requestGenerator(batched_image_data, input_name, output_name, dtype):
    inputs = []
    inputs.append(grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = []
    for name in output_name:
        outputs.append(grpcclient.InferRequestedOutput(name))

    yield inputs, outputs
