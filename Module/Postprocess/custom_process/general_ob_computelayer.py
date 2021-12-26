from Module.builder import POSTPROCESSES
from .general_ob_modellayer import General_ob_modellayer
from Module.Postprocess.common.compute_layer import stackSKUProcRaw, stackSKUProcSpecial, filterSKUByLayer, \
    filterSKUByLayerFine, ConvertLayers, skuStatus, SKU, getskuList


@POSTPROCESSES.register_module()
class General_ob_computelayer(General_ob_modellayer):
    def __init__(self, stack_config, layer_config):
        self.stack_config = stack_config
        self.layer_config = layer_config

    def merge_sku_cjfc(self, result_processed, width, height):
        # skudata = self.format_bbox_result(skulist, width, height)
        imageinfo = {"width": width, "height": height, "distance": 1.0, "isVision": 0}
        skulist = result_processed["sku"]

        skukit = SKU(skulist, imageinfo)
        skudata = skukit.getskudict()

        if self.stack_config["Enable"]:
            if self.stack_config["Func"] == "Raw":
                skudata = stackSKUProcRaw(skudata)
            elif self.stack_config["Func"] == "Special":
                skudata = stackSKUProcSpecial(skudata, self.stack_config)

        skuslist = getskuList(skudata)

        if self.layer_config["Func"] == "Fine":
            layers = filterSKUByLayerFine(skuslist, self.layer_config, imageinfo)
        else:
            layers = filterSKUByLayer(skuslist, self.layer_config, imageinfo)

        converter = ConvertLayers(imageinfo, layers)
        skuresult = converter.convert()
        skus = skuresult['boxInfo']
        skuresult['skuStat'] = skuStatus(skus)
        return skuresult

    def cjfc_process(self, cjfclist):
        return cjfclist
