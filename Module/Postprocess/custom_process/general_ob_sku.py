from Module.builder import POSTPROCESSES
from .general_ob_modellayer import General_ob_modellayer


@POSTPROCESSES.register_module()
class General_ob_sku(General_ob_modellayer):
    def merge_sku_cjfc(self, result_processed, width, height):
        skulist = result_processed["sku"]
        return self.format_bbox_result(skulist, width, height)
