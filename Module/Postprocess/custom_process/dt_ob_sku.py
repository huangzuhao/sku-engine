from Module.builder import POSTPROCESSES
from .general_ob_modellayer import General_ob_modellayer


@POSTPROCESSES.register_module()
class Dt_ob_sku(General_ob_modellayer):
    def merge_sku_cjfc(self, result_processed, width, height):
        skulist = result_processed["sku"]
        return self.process_dt_result(skulist)

    def process_dt_result(self, result_list):
        result = {}
        if len(result_list) > 0:
            for sku in result_list:
                if sku[5] == 'signboard':
                    result['resultlist'] = [u'店面照']
                    result['resultdetaillist'] = [{'class': u'店面照', 'confidence': sku[4]}]
                    break
                else:
                    result['resultlist'] = [u'非店面照']
                    result['resultdetaillist'] = [{'class': u'非店面照', 'confidence': sku[4]}]
        else:
            result['resultlist'] = [u'非店面照']
            result['resultdetaillist'] = [{'class': u'非店面照', 'confidence': 0.99}]
        return result
