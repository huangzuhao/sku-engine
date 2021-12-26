from Module.builder import POSTPROCESSES
from .general_ob_modellayer import General_ob_modellayer
from .general_ob_sku import General_ob_sku
from Module.Postprocess.common.common_method import between_class_nms_by_score


@POSTPROCESSES.register_module()
class Dz_ob_modellayer(General_ob_modellayer):
    def between_class_nms(self, skulist):
        return between_class_nms_by_score(skulist, excludelist=["yl_bkl_wl"])


@POSTPROCESSES.register_module()
class Dz_ob_sku(General_ob_sku):
    def between_class_nms(self, skulist):
        return between_class_nms_by_score(skulist, excludelist=["yl_bkl_wl"])
