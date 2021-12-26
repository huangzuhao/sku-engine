from Module.builder import POSTPROCESSES
from .general_ob_base import General_ob_base
from collections import Counter, OrderedDict
from shapely.geometry import Polygon
from copy import deepcopy
import numpy as np
from Module.Postprocess.common.common_method import delete_special_sku, between_class_nms_by_score


@POSTPROCESSES.register_module()
class General_ob_modellayer(General_ob_base):

    def merge_sku_cjfc(self, result_processed, width, height):
        #     "sku": [[0.100, 0.200, 300, 400, 0.93, "sku1"],
        #                [100, 200, 300, 400, 0.93, "sku2"],
        #                [100, 200, 300, 400, 0.93, "sku3"]],
        #     "cjfc": [[100, 200, 300, 400, 0.93, "sku1",[[123.0,123.0],[126.0,126.0],145.0,145.0,169.1,169.3]],
        #              [100, 200, 300, 400, 0.93, "sku2",[123.0,123.0,126.0,126.0,145.0,145.0,169.1,169.3]],
        #              [100, 200, 300, 400, 0.93, "sku3",[123.0,123.0,126.0,126.0,145.0,145.0,169.1,169.3]]]}
        # 过滤保留主场景，及SKU分层
        skulist = result_processed["sku"]
        cjfclist = result_processed["cjfc"]

        layerdata = self.format_mask_result(cjfclist, width, height)

        skudata = self.format_bbox_result(skulist, width, height)

        sku_result = self.layer_detect_multi_scene(skudata, layerdata)

        return sku_result

    def layer_detect_multi_scene(self, skudata, layerdata):
        scenes = []
        layers = []
        for shape in layerdata['shapes']:
            label = shape['label']
            if len(shape['points']) > 2:
                ply = Polygon(shape['points'])
                shape['polygon'] = ply
                if 'cj' in label:
                    scenes.append(shape)
                if 'layer' in label:
                    layers.append(shape)

        # 如果图像无分层识别结果，则所有SKU当作一层
        if not len(layers):
            skudata['layerNum'] = 1
            for sku in skudata['boxInfo']:
                sku['layer'] = '1'
            return skudata

        # 删除重合的层
        layers = self.delete_iou_layer(layers)

        # 根据层的质心是否在主场景内，找出每个场景内的所有层
        layer2scene = OrderedDict()
        for layer in layers:
            lply = layer['polygon']
            centroid = lply.centroid
            layer['centroid'] = np.concatenate(centroid.coords.xy)

            hit = False
            for scene in scenes:
                sname = scene['label']
                sply = scene['polygon']
                # 如果层的质心落于场景内，则保留该层
                if sply.contains(lply.centroid):
                    if sname not in layer2scene:
                        layer2scene[sname] = [layer]
                    else:
                        layer2scene[sname].append(layer)
                    # 每一层只隶属于一个场景
                    hit = True
                    break

            if not hit or not len(scenes):
                if 'others' not in layer2scene:
                    layer2scene['others'] = [layer]
                else:
                    layer2scene['others'].append(layer)

        # 给所有层在各自场景内排序
        sorted_layers = []
        layerNum = 0
        for sname, layers in layer2scene.items():
            layerNum = max(layerNum, len(layers))
            inds = np.argsort([layer['centroid'][1] for layer in layers])
            for i, idx in enumerate(inds, 1):
                layer = deepcopy(layers[idx])
                layer['scene'] = sname
                layer['layer'] = str(i)
                sorted_layers.append(layer)

        # 计算SKU与层的IOU，将SKU归类到层
        # N*M array, N skus, M layers
        iouM = self.iou_sku_layer(skudata, sorted_layers)

        boxInfo = []
        for i, iou in enumerate(iouM):
            idxs = np.where(iou == np.max(iou))[0].tolist()
            closest_ind = 0
            Hdistance = 1
            if len(idxs) > 1:
                for idx in idxs:
                    sku_ymin = skudata['boxInfo'][i]['location']['ymin']
                    sku_ymax = skudata['boxInfo'][i]['location']['ymax']
                    sku_cy = (sku_ymin + sku_ymax) / 2
                    layer_cy = sorted_layers[idx]['centroid'][1] / skudata['imageInfo']['height']
                    distance = abs(sku_cy - layer_cy)
                    if distance < Hdistance:
                        Hdistance = distance
                        closest_ind = idx
            else:
                closest_ind = idxs[0]
            iou = iou[closest_ind]
            sku = skudata['boxInfo'][i]
            # 有些情况层分割不准，没能包含SKU，则将SKU层数置为1
            layer = '1' if iou == 0 else sorted_layers[closest_ind]['layer']
            # del sku['layerOut']  # 后续用不到
            sku['layer'] = layer
            boxInfo.append(sku)

        # 重新计算skuStat
        skuStat = [sku['skuName'] for sku in boxInfo]
        skuStat = dict(Counter(skuStat))
        skuStat = [dict(count=count, skuName=skuName) for skuName, count in skuStat.items()]

        skudata['boxInfo'] = boxInfo
        skudata['skuStat'] = skuStat
        skudata['layerNum'] = layerNum

        return skudata

    def iou_sku_layer(self, skudata, layers):
        # 解析图像信息
        imageInfo = skudata['imageInfo']
        imageWidth = imageInfo['width']
        imageHeight = imageInfo['height']

        # layer 包含Polygon字段对象
        iou = np.zeros((len(skudata['boxInfo']), len(layers)), dtype=np.float)
        for i, sku in enumerate(skudata['boxInfo']):
            loc = sku['location']
            xmin = loc['xmin'] * imageWidth
            ymin = loc['ymin'] * imageHeight
            xmax = loc['xmax'] * imageWidth
            ymax = loc['ymax'] * imageHeight
            points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
            sku = Polygon(points)
            # 以SKU的面积作为计算IOU的分母
            union = sku.area
            for j, layer in enumerate(layers):
                layer = layer['polygon'].buffer(0)  # buffer是针对某些情况下的异常处理
                inter = layer.intersection(sku).area
                iou[i, j] = inter / union

        return iou

    def delete_iou_layer(self, layers):
        length = len(layers)
        if length > 0:
            del_list = []
            for i in range(length):
                for j in range(i + 1, length):
                    layer1 = layers[i]['polygon'].buffer(0)
                    layer2 = layers[j]['polygon'].buffer(0)
                    minarea = min(layer1.area, layer2.area)
                    if minarea < 1.0:
                        minarea = 1.0
                    interarea = layer1.intersection(layer2).area
                    iou = interarea / minarea

                    if iou > 0.7:
                        if layer1.area < layer2.area:
                            del_list.append(i)
                        else:
                            del_list.append(j)
            del_list = list(set(del_list))
            del_list.sort(reverse=True)
            for i in del_list:
                del layers[i]
            return layers
        else:
            return []

    def delete_fake(self, skulist):
        skulist_nofake = delete_special_sku(skulist)
        return skulist_nofake

    def cjfc_process(self, cjfclist):
        return cjfclist

    def between_class_nms(self, skulist):
        sku_after_nms = between_class_nms_by_score(skulist)
        return sku_after_nms

    def boxes_process(self, boxeslist):
        return boxeslist

    def format_bbox_result(self, result_list, width, height):
        boxInfo = []
        skuStat = {}
        for sku in result_list:
            boxInfo.append(
                {
                    'layer': '',
                    'skuName': sku[5],
                    'location': {'xmin': round(sku[0], 4), 'ymin': round(sku[1], 4), 'xmax': round(sku[2], 4),
                                 'ymax': round(sku[3], 4)},
                    'score': round(sku[4], 4)
                }
            )
            if sku[5] in skuStat:
                skuStat[sku[5]] += 1
            else:
                skuStat[sku[5]] = 1
        stat = []
        for k, elem in skuStat.items():
            stat.append({'count': elem, 'skuName': k})
        imageinfo = {"width": width, "height": height, "distance": 1.0, "isVision": 0}

        result = {'boxInfo': boxInfo, 'skuStat': stat, 'layerOut': [], 'imageInfo': imageinfo}

        return result

    def format_mask_result(self, mask_list, width, height):
        mask_dict = {}
        mask_dict["imageHeight"] = height
        mask_dict["imageWidth"] = width
        mask_dict["imageData"] = None

        mask_dict["lineColor"] = [0, 255, 0, 128]
        mask_dict["fillColor"] = [255, 0, 0, 128]
        mask_dict["flags"] = {}

        shapes = []
        labelnames = {}

        for point in mask_list:
            shape_dict = {}
            shape_dict["line_color"] = None
            shape_dict["fill_color"] = None
            if point[5] in labelnames:
                labelnames[point[5]] += 1
            else:
                labelnames[point[5]] = 1
            shape_dict["label"] = point[5] + "-" + str(labelnames[point[5]])
            shape_dict["shape_type"] = "polygon"
            shape_dict["flags"] = {}
            shape_dict["points"] = point[-1]

            shapes.append(shape_dict)

        mask_dict["shapes"] = shapes
        return mask_dict
