from Module.builder import POSTPROCESSES
from .general_ob_computelayer import General_ob_computelayer
from Module.Postprocess.common.common_method import delete_special_sku, between_class_nms_by_score

# @POSTPROCESSES.register_module()
# class Hn_ob_computelayer(General_ob_computelayer):
#
#     def between_class_nms(self, skulist):
#         skulist = delete_special_sku(skulist, conditionlist=["anjbj", "101bj"])
#         length = len(skulist)
#         if length > 0:
#             for i in range(length):
#                 if skulist[i][5] == 'a_hn_anj':
#                     skulist[i][5] = 'a_hn_101'
#             skulist = between_class_nms_by_score(skulist)
#             return skulist
#         else:
#             return []
