from Module.builder import POSTPROCESSES
from Module.Postprocess.common.common_method import merge_sku_others
from abc import ABCMeta, abstractmethod


# @POSTPROCESSES.register_module()
class General_ob_base(metaclass=ABCMeta):
    def __init__(self):
        pass

    def process(self, resultdata):
        # resultdata = {
        #    "imageInfo": {"width": 1600, "height": 1200},
        #     "sku": [[0.100, 0.200, 0.300, 0.400, 0.93, "sku1"],
        #             [0.100, 0.200, 0.300, 0.400, 0.93, "sku2"],
        #             [0.100, 0.200, 0.300, 0.400, 0.93, "sku3"]],
        #     "others": [[0.100, 0.200, 0.300, 0.400, 0.93, "daiding_101"],
        #                [0.100, 0.200, 0.300, 0.400, 0.93, "sku2"],
        #                [0.100, 0.200, 0.300, 0.400, 0.93, "sku3"]],
        #     "cjfc": [[0.1, 0.2, 0.3, 0.4, 0.93, "sku1",[[123.0,123.0],[126.0,126.0],[145.0,145.0],[169.1,169.3]]],
        #             [0.1, 0.2, 0.3, 0.4, 0.93, "sku1",[[123.0,123.0],[126.0,126.0],[145.0,145.0],[169.1,169.3]]],
        #              [0.1, 0.2, 0.3, 0.4, 0.93, "sku1",[[123.0,123.0],[126.0,126.0],[145.0,145.0],[169.1,169.3]]]]}

        width = resultdata["imageInfo"]["width"]
        height = resultdata["imageInfo"]["height"]
        sku_data = []
        others_data = []
        cjfc_data = []
        boxes_data = []

        if "sku" in resultdata:
            sku_data = resultdata["sku"]
        if "others" in resultdata:
            others_data = resultdata["others"]
        if "cjfc" in resultdata:
            cjfc_data = resultdata["cjfc"]
        if "boxes" in resultdata:
            boxes_data = resultdata["boxes"]

        skulist_nofake = self.delete_fake(sku_data)

        # skulist_clean = between_class_nms(skulist_nofake)
        skulist_clean = self.between_class_nms(skulist_nofake)

        cjfclist_clean = self.cjfc_process(cjfc_data)

        sku_others_list = merge_sku_others(skulist_clean, others_data)

        boxeslist_clean = self.boxes_process(boxes_data)

        result_processed = {}
        result_processed["sku"] = sku_others_list
        result_processed["boxes"] = boxeslist_clean
        result_processed["cjfc"] = cjfclist_clean

        # sku_cjfc_reslut = self.merge_sku_cjfc(sku_others_list, cjfclist_clean, width, height)
        sku_cjfc_reslut = self.merge_sku_cjfc(result_processed, width, height)

        return sku_cjfc_reslut

    @abstractmethod
    def merge_sku_cjfc(self, result_processed, width, height):
        pass

    @abstractmethod
    def cjfc_process(self, cjfclist):
        pass

    @abstractmethod
    def delete_fake(self, skulist):
        pass

    @abstractmethod
    def between_class_nms(self, skulist):
        pass

    @abstractmethod
    def boxes_process(self, boxeslist):
        pass
