import copy

from Module.builder import POSTPROCESSES
from .general_ob_modellayer import General_ob_modellayer
from .general_ob_computelayer import General_ob_computelayer
from .general_ob_sku import General_ob_sku
from Module.Postprocess.common.common_method import delete_special_sku, between_class_nms_by_score
from shapely.geometry import Polygon
import cv2
import numpy as np
from scipy.spatial.distance import pdist
from functools import reduce
from Module.Postprocess.common.compute_layer import stackSKUProcRaw, stackSKUProcSpecial, filterSKUByLayer, \
    filterSKUByLayerFine, ConvertLayers, skuStatus, SKU, getskuList

cj_map = {'ty_cj_others': 0, 'ty_cj_hj': 1, 'ty_cj_tg': 2, 'ty_cj_clj': 3, 'ty_cj_mit': 4, 'ty_cj_gx': 5,
          'ty_cj_lsbx': 6, 'ty_lsbx_cj': 7, 'ty_cj_ryj': 8, 'ty_ryj_cj': 9, 'ty_cj_wsbg': 10,
          'ty_cj_clgj': 11, 'ty_cj_lfg': 12, 'ty_cj_lsbx_ty': 13, 'ty_cj_logo_ty_fake': 14, 'ty_cj_ryj_logo': 15,
          'ty_cj_ryj_fake': 16, 'ty_cj_logo_ty': 17}  # 'ty_cj_null':13,       #新增了近距离空场景，placeType=13


@POSTPROCESSES.register_module()
class Ty_ob_modellayer(General_ob_modellayer):
    def cjfc_process(self, cjfclist):
        points, double_lsbx, soc_mlsbx = cj_result_process(cjfclist)
        res = get_mask_dict(points)
        # 计算res中的ty_cj_lsbx数、ty_lsbx_cj数
        shLsbxNum, tyLsbxNum = 0, 0
        for shape in res['shapes']:
            if shape['label'].split('-')[0] == 'ty_cj_lsbx':
                shLsbxNum += 1
            elif shape['label'].split('-')[0] == 'ty_lsbx_cj':
                tyLsbxNum += 1
        res_d_lsbx = get_mask_dict(double_lsbx, cj_mode='doublelsbx', lsbxNum=tyLsbxNum)
        res_sh_lsbx = get_mask_dict(soc_mlsbx, cj_mode='doublelsbx', lsbxNum=shLsbxNum)
        res['double_lsbx'] = res_d_lsbx  # 增加键值
        res['soc_mlsbx'] = res_sh_lsbx  # 增加键值
        return res

    def merge_sku_cjfc(self, result_processed, width, height):
        imgInfo = {"width": width, "height": height, "distance": 1.0, "isVision": 0}
        skulist = result_processed["sku"]
        skudata = self.format_bbox_result(skulist, width, height)
        # skudata, skulist, imgInfo, thres, skus = skuDetect(skuDet,bottleDet, img)
        # skudata = fix_json(skudata)
        try:
            # layerdata = maskLayerDetect(cjfcDet, img)  ## 这里调用 cj_result_process -> procces_lsbx
            layerdata = result_processed["cjfc"]
            scene_info = getSceneInfo(layerdata)  # 加入了全场景 #
            # pdb.set_trace()
            if scene_info['sceneSetNum'] == 1:  # 如果没有识别到场景和分层信息(只有others)
                # _, skudata, _ = postProcess(skudata, imgInfo, thres, skulist)
                results = fomart_data_without_cj1(skudata)
            else:
                # 1.处理割箱场景
                # pdb.set_trace()
                gx_info, layerdata = getCjinfo(layerdata, ['ty_cj_gx'])
                if gx_info:
                    # gxskudata, gxskulist, imgInfo, thres, skus = skuDetect(gxDet,skulist, img)
                    # gxskudata = fix_json(gxskudata)
                    boxeslist = result_processed["boxes"]
                    gxskudata = self.format_bbox_result(boxeslist, width, height)

                    # 20210811 把gxskudata和skudata合并成skudata
                    skudata['boxInfo'] += gxskudata['boxInfo']
                    skudata['skuStat'] += gxskudata['skuStat']
                    # skulist += gxskulist
                gx_skudata, skudata, gx_cj_sku = postProcess(skudata, imgInfo, scene=gx_info)
                gx_skudata = skuInfoFormat(gx_skudata)
                gx_result = fomart_data(gx_skudata, scene_info, gx_cj_sku)

                # 2.处理不需分层的场景（地堆，卧式冰柜）
                mit_info, layerdata = getCjinfo(layerdata, ['ty_cj_mit', 'ty_cj_wsbg', 'ty_cj_others'])
                mit_skudata, skudata, mit_cj_sku = postProcess(skudata, imgInfo, scene=mit_info)
                mit_skudata = skuInfoFormat(mit_skudata)
                mit_result = fomart_data(mit_skudata, scene_info, mit_cj_sku)

                # 3. 处理需要精确分层的场景
                # pdb.set_trace() #这个函数看下
                res, res_cj = layerDetectMask(skudata, layerdata, imgInfo)  # 调用layer_detect对层排序（加待定PPPP）
                # pdb.set_trace();print('【res】')
                # if not res_cj:
                if (not res_cj and not gx_result and not mit_result):
                    # _, skudata, _  = postProcess(skudata, imgInfo, thres, skulist)  #<-bug
                    result = fomart_data_without_cj1(skudata)  # 没有场景的sku当成了全场景
                else:
                    # elif (res_cj and not gx_result and not mit_result):
                    res = skuInfoFormat(res)
                    result = fomart_data(res, scene_info, res_cj=res_cj)
                # else:
                #     result = []
                # 合并上述三种处理后的结果
                results = gx_result + mit_result + result

        except Exception as e:
            print(repr(e))
            # _, skudata, _  = postProcess(skudata, imgInfo, thres, skulist)
            results = fomart_data_without_cj1(skudata)

        # imgInfo['direction'] = direction
        for place in results:  # 修改实例表示为语义表示 方法
            place['placeSegname'] = place['placeSegname'].split('-')[0]
            for box in place['boxInfo']:
                box['skuName'] = box['skuName'].split('-')[0]
        results = {'dataInfo': results, 'imageInfo': imgInfo}
        # pdb.set_trace();print('- end result -')
        results = closePicture(results)  # 近距离拍照
        return results


# def postProcess(skudata, imgInfo, thres, skuslist=None, scene=None):
#     # step5: To put the skus into different layers using the stacked skulist
#     # pdb.set_trace()
#     cj_sku = {}
#     new_skudata = {'boxInfo': []}
#     if skuslist:
#         proc = LayerProcess(imgInfo)
#         layerDict, layers = proc.filterSKUByLayerFine(skuslist, thres)
#         # else:
#         # layerDict,layers=proc.filterSKUByLayer(skuslist,thres)
#
#         # step6: To calculate the layerLine and get the final results.
#         converter = ConvertLayers(imgInfo, layers)
#         skudata = converter.convert()
#     img_h = imgInfo['height']
#     img_w = imgInfo['width']
#
#     if scene:
#
#         sku_info, skudata['boxInfo'] = skuInstance(skudata['boxInfo'])
#
#         # # 场景信息
#         cj_info = scene
#         skuClass = [ele['skuName'] for ele in sku_info]
#         sku_coords = compute_all_centroid(sku_info, img_h, img_w)  # 计算所有sku的质心
#         # sku 分配到各场景
#         cj_sku = {}
#         for cj, point in cj_info.items():
#             if 'cj' in cj:
#                 temp = []
#                 for sku in skuClass:
#                     if isInsidePolygon(sku_coords[sku], point):
#                         temp.append(sku)
#                 cj_sku[cj] = temp
#
#         # 把已分配的SKU从skulist和skudata中删除
#         # _, skuslist = splitSkuList(skuslist, cj_sku)
#
#         new_skudata = skudata.copy()
#         new_skudata['boxInfo'], skudata['boxInfo'] = splitSkuList(skudata['boxInfo'], cj_sku)
#         skus = new_skudata['boxInfo']
#         new_skudata['skuStat'] = skuStatus(skus)
#
#     for box in skudata['boxInfo']:
#         box['skuName'] = box['skuName'].split('-')[0]
#     skus = skudata['boxInfo']
#     skudata['skuStat'] = skuStatus(skus)
#
#     return new_skudata, skudata, cj_sku


def skuInstance(skulist):
    sku_class = [ele['skuName'] for ele in skulist]
    sku_class = set(sku_class)
    sku_label_count = {}
    for sku_label in sku_class:
        sku_label_count.update({sku_label: 0})
    sku_info = []
    for ele in skulist:
        ele_info = ele
        sku_label = ele['skuName']
        sku_label_count[sku_label] += 1
        ele_info['skuName'] = sku_label + '-' + str(sku_label_count[sku_label])
        sku_info.append(ele_info)
    return sku_info, skulist


def skuStatus(skus):
    skuStat = {}
    for sku in skus:
        if sku['skuName'] in skuStat:
            skuStat[sku['skuName']] += 1
        else:
            skuStat[sku['skuName']] = 1

    stat = []
    for k, elem in skuStat.items():
        stat.append({
            "count": elem,
            "skuName": k
        })

    return stat


def splitSkuList(skulist, cj_sku):
    new_shapes = []
    skulist_bak = skulist.copy()
    for cj, sku in cj_sku.items():
        for ele in skulist:
            if ele['skuName'] in sku:
                new_shapes.append(ele)
                skulist_bak.remove(ele)
    return new_shapes, skulist_bak


# def compute_all_centroid(all_info, img_h, img_w):
#     ''' 计算所有SKU的中心点坐标
#     Args:
#        all_info: 所有sku的信息
#        img_h, img_w: 图像的高度和宽度
#     Returns:
#        centroid_dict: 字典, {'skuname-i':[x1, y1], 'skuname-i':[x1, y1]}
#     '''
#     centroid_dict = {}
#     for ele in all_info:
#         point = point_trans(ele, img_w, img_h)
#         # point = ele['points']
#         centroid = compute_centroid(point)
#         centroid_dict[ele['skuName']] = centroid
#     return centroid_dic


def isInsidePolygon(pt, poly):
    '''采用射线法判断一个点似乎否在一个东边形内
    Args;
       pt: 点
       poly: 多边形

    Returns:
       flag: True/False True表示该点在多边形内
    '''
    flag = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
            if (pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]):
                flag = not flag
        j = i
    return flag


def get_mask_dict(points, cj_mode='normal', lsbxNum=0):
    """ 基于轮廓点信息和轮廓的名称保存为json格式文件
    update: 2020/12/16 by linmuxing
    log：增加labels入参，用于更新场景后处理后的实例编码
    Args:

        points: 轮廓点及轮廓名称列表

    Return:
        mask_dict:
    """
    mask_dict = {}
    # mask_dict["imageHeight"] = img.shape[0]
    # mask_dict["imageWidth"]  = img.shape[1]
    mask_dict["imageData"] = None
    mask_dict["lineColor"] = [0, 255, 0, 128]
    mask_dict["fillColor"] = [255, 0, 0, 128]
    mask_dict["flags"] = {}
    shapes = []
    labelSetCount = {}

    if cj_mode == 'normal':
        for point in points:
            shape_dict = {}
            shape_dict["line_color"] = None
            shape_dict["fill_color"] = None
            if point[5] in labelSetCount:
                labelSetCount[point[5]] += 1
            else:
                labelSetCount[point[5]] = 1

            shape_dict["label"] = point[5] + "-" + str(labelSetCount[point[5]] + lsbxNum)
            shape_dict["shape_type"] = "polygon"
            shape_dict["flags"] = {}
            shape_dict["points"] = point[-1]
            shapes.append(shape_dict)

        mask_dict["shapes"] = shapes
    elif cj_mode == 'doublelsbx':
        for ind, point_double in enumerate(points):
            shapes.append([])
            for point in point_double:
                shape_dict = {}
                shape_dict["line_color"] = None
                shape_dict["fill_color"] = None
                if point[5] in labelSetCount:
                    labelSetCount[point[5]] += 1
                else:
                    labelSetCount[point[5]] = 1

                shape_dict["label"] = point[5] + "-" + str(labelSetCount[point[5]] + lsbxNum)
                shape_dict["shape_type"] = "polygon"
                shape_dict["flags"] = {}
                shape_dict["points"] = point[-1]
                shapes[ind].append(shape_dict)
        mask_dict["shapes"] = shapes

    return mask_dict


def get_mask_dict_cjfc(img, points, labelnameSet, imgpath, jsonname):
    """ 基于轮廓点信息和轮廓的名称保存为json格式文件
    update: 2020/12/16 by linmuxing
    log：增加labels入参，用于更新场景后处理后的实例编码
    Args:
        img: opencv读入的图像数组
        points: 轮廓点及轮廓名称列表
        labelnameSet: 标签元组
    Return:
        mask_dict:

    """
    mask_dict = {}
    mask_dict["imagePath"] = imgpath.split('/')[-1]
    mask_dict["imageHeight"] = img.shape[0]
    mask_dict["imageWidth"] = img.shape[1]
    mask_dict["imageData"] = None

    mask_dict["lineColor"] = [0, 255, 0, 128]
    mask_dict["fillColor"] = [255, 0, 0, 128]
    mask_dict["flags"] = {}

    shapes = []
    labelSetCount = {}
    for labelname in labelnameSet:
        labelSetCount.update({labelname: 0})
    for point in points:
        # print(point)
        # shape_dict = {}
        # shape_dict["line_color"] = None
        # shape_dict["fill_color"] = None
        # if 'layer' not in point[-2] and point[-2] in labelnameSet:
        if point[-2] in labelnameSet:
            shape_dict = {}
            shape_dict["line_color"] = None
            shape_dict["fill_color"] = None

            labelSetCount[point[-2]] += 1
            shape_dict["label"] = point[-2]  # + "-" + str(labelSetCount[point[-2]])
            shape_dict["shape_type"] = "polygon"
            shape_dict["flags"] = {}
            shape_dict["points"] = point[:-2]

            shapes.append(shape_dict)

    mask_dict["shapes"] = shapes
    # with open(jsonname, "w") as f:
    #     json.dump(mask_dict, f, cls=NpEncoder, ensure_ascii=False, indent=4, separators=(',',':'))

    return mask_dict


# 冰柜、热饮机后处理
def cj_result_process(result_list):
    # pdb.set_trace();print('=cj_process=') #result_list = cj_nms(result_list)
    sku_list = ["ty_cj_lsbx_ty", "ty_cj_lsbx", "ty_cj_logo_ty_fake", "ty_cj_logo_ty", 'ty_cj_ryj', 'ty_cj_ryj_logo',
                'ty_cj_ryj_fake', 'ty_cj_lsbx_sh']  # 加了ty_cj_lsbx_sh
    list_cj = []
    if len(result_list) > 0:
        list_lsbx_ty = get_sku_box("ty_cj_lsbx_ty", result_list)  # 返回结果中的"ty_cj_lsbx_ty"
        list_lsbx = get_sku_box("ty_cj_lsbx", result_list)
        list_logo_fake = get_sku_box("ty_cj_logo_ty_fake", result_list)
        list_logo = get_sku_box("ty_cj_logo_ty", result_list)

        # 直接过滤list_lsbx中面积小于50%的lsbx? 20210902
        # list_lsbx = [lsbx for lsbx in list_lsbx if if_regular_lsbx(lsbx)] 改为下面
        # 加入list_lsbx之间比较小的去除 20210917
        list_lsbx = del_half_lsbx(list_lsbx)

        list_ryj = get_sku_box("ty_cj_ryj", result_list)
        list_ryj_fake = get_sku_box("ty_cj_ryj_fake", result_list)
        list_ryj_logo = get_sku_box("ty_cj_ryj_logo", result_list)

        # 20210809 "social double lsbx"
        list_mlsbx_sh = get_sku_box("ty_cj_lsbx_sh", result_list)  # 返回结果中的"ty_cj_lsbx_sh" 双开门社会冰箱

        list_cj = get_sku_box(sku_list, result_list)  # layer和gx、hj等场景
        list_cj = del_small_layer(list_cj)
        # pdb.set_trace()

        # 需要在处理前提前进行如下nms：
        list_lsbx_ty = cj_nms(list_lsbx_ty)  # 这里增加了 tylsbx - nms
        list_mlsbx_sh = cj_nms(list_mlsbx_sh)
        list_lsbx = lsbx_nms(list_lsbx)
        list_lsbx = cj_nms(list_lsbx)  # 这里增加了 lsbx - nms
        # list_cj = layer_nms(list_cj)        #这里增加了 layer - nms

        proceed_lsbx, double_lsbx, soc_mlsbx = procces_lsbx(list_lsbx_ty, list_mlsbx_sh, list_lsbx, list_logo,
                                                            list_logo_fake, list_cj)  # 这里多传入，返回double_lsbx、soc_dlsbx结果
        list_cj = procces_ryj(list_ryj, list_ryj_logo, list_ryj_fake, list_logo, list_logo_fake,
                              proceed_lsbx)  # 这里要传入proceed_lsbx，需检查

    list_cj = cj_nms(list_cj)
    # pdb.set_trace();print('=cj_process_result=')
    return list_cj, double_lsbx, soc_mlsbx


def get_sku_box(sku_name, result_list):
    if isinstance(sku_name, list):
        return [boxlist
                for boxlist in result_list if boxlist[-2] not in sku_name]  # layer?
    else:
        return [boxlist
                for boxlist in result_list if boxlist[-2] == sku_name]


def del_half_lsbx(list_lsbx):
    '''删除露出面积小于50%的lsbx框'''
    if not list_lsbx:
        return list_lsbx
    lsbx_area = {}
    list_lsbx_result = []
    for i in range(len(list_lsbx)):
        # lsbx_area[i] = cv2.contourArea(np.array(list_lsbx[i][:-2]))
        lsbx_area[i] = cv2.contourArea(np.array(list_lsbx[i][-1]))
    MAXAREA = sorted(lsbx_area.values(), reverse=True)[0]
    for i in range(len(list_lsbx)):
        points = list_lsbx[i]
        # 1.过滤三角形的lsbx
        if len(points[-1]) < 4:  # points只有3个点 (p1,p2,p3,label,score)
            continue
        # 2.过滤高宽比超过阈值的
        coordx = [c[0] for c in points[-1]]
        coordy = [c[1] for c in points[-1]]
        h, w = max(coordy) - min(coordy), max(coordx) - min(coordx)
        hw_ratio = h / w
        # print('hw_ratio', hw_ratio)
        if hw_ratio > 5.5 or hw_ratio < 0.182:  # 细长形过滤掉
            # if hw_ratio > 6 or hw_ratio < 0.167: #细长形过滤掉
            continue
        # 3.过滤掉面积小于其外接矩形面积0.59倍的
        # print('area/(h*w)', lsbx_area[i] / (h * w))
        if lsbx_area[i] / (h * w) < 0.553:  # 这里可能需优化 0.59
            continue
        # 4.过滤掉面积小于图中最大lsbx面积0.4倍的
        # print('area/max_lsbx', lsbx_area[i] / MAXAREA)
        if lsbx_area[i] / MAXAREA < 0.38:  # 0.418
            continue
        list_lsbx_result.append(points)
        # print(' 保留此lsbx')
    return list_lsbx_result


def del_small_layer(list_cj):
    '''删除极小没有意义的层，保证lsbx合格性判断'''
    layer_areas = {}  # 层的面积
    layer_box_areas = {}  # 层的外接矩形面积
    for i in range(len(list_cj)):
        if list_cj[i][-2] == 'layer':
            coordx = [c[0] for c in list_cj[i][-1]]
            coordy = [c[1] for c in list_cj[i][-1]]
            layer_box_areas[i] = (max(coordy) - min(coordy)) * (max(coordx) - min(coordx))
    if not layer_areas.values():
        return list_cj
    MAXAREA = max(layer_areas.values())
    list_cj_result = []
    for i in range(len(list_cj)):
        if list_cj[i][-2] == 'layer':
            if layer_areas[i] / MAXAREA < 0.032 and layer_areas[i] / layer_box_areas[i] < 0.6:
                # print('del a small layer');
                continue
        list_cj_result.append(list_cj[i])
    return list_cj_result


# 类间极非大抑制
def cj_nms(input_list):
    """
    计算sku类别间的非极大抑制
    update: 2020/12/16 by linmuxing
    Args:
        input_list：处理后的mask轮廓点[[x1,y1],[x2,y2],...,'sku',score]

    Return:
       temp_list: 类间nms后的识别结果

    """
    # temp_list = del_noneed(input_list)
    length = len(input_list)
    if length > 1:
        del_list = []
        for i in range(length):
            for j in range(i + 1, length):
                # print(input_list[i])
                # iou = calculate_iou(input_list[i][:-2],input_list[j][:-2])
                iou = calculate_iou(input_list[i][-1], input_list[j][-1])
                if iou > 0.8 and 'layer' not in set([input_list[i][-2], input_list[j][-2]]):  # layer不参与nms
                    group = set([input_list[i][-2], input_list[j][-2]])
                    if group == {'ty_cj_hj', 'ty_cj_lsbx', 'ty_cj_ryj', 'ty_lsbx_cj', 'ty_ryj_cj'} or group == {
                        'ty_cj_hj', 'ty_cj_tg'}:
                        if input_list[i][-2] == 'ty_cj_hj':
                            del_list.append(i)
                        else:
                            del_list.append(j)
                    else:
                        # print('score',input_list[i][-1])
                        # if input_list[i][-1] < input_list[j][-1]: #根据score
                        if input_list[i][-3] < input_list[j][-3]:  # 根据score
                            del_list.append(i)
                        else:
                            del_list.append(j)
        del_list = list(set(del_list))
        del_list.reverse()
        temp = input_list.copy()
        for i in del_list:
            temp.remove(input_list[i])
            #   del input_list[i]
        return temp
    else:
        return input_list


def lsbx_nms(list_lsbx):
    '''专用于处理双门场景中，一个ty_cj_lsbx包住两个ty_cj_lsbx的情况'''
    if len(list_lsbx) > 1:
        del_list = []
        for i in range(len(list_lsbx)):
            encase_num = 0
            for j in range(len(list_lsbx)):
                if i == j:
                    continue
                # iou = calculate_iou(list_lsbx[i][:-2], list_lsbx[j][:-2])
                iou = calculate_iou(list_lsbx[i][-1], list_lsbx[j][-1])
                if iou > 0.8:
                    encase_num += 1
            if encase_num > 1:
                del_list.append(list_lsbx[i])
        for lsbx in del_list:
            if lsbx in list_lsbx:
                list_lsbx.remove(lsbx)
    return list_lsbx


def procces_lsbx(list_lsbx_ty, list_mlsbx_sh, list_lsbx, list_logo, list_logo_fake, list_cj):
    """
    立式冰箱后处理逻辑
    update: 2020/12/16 by linmuxing
    Args:
        list_lsbx_ty: 立式冰箱（含冰柜头）识别结果
        list_mlsbx_sh: 社会立式冰箱
        list_lsbx: 立式冰箱识别结果
        list_logo: 统一立式冰箱冰柜头识别结果
        list_logo_fake: 立式冰箱层间的统一字样logo
        list_cj: 其他场景识别结果
    Return:
        list_cj: 其他场景和处理后的立式冰箱场景合并结果
    """
    '''
    list_lsbx_ty: [[[227, 42], [240, 1361], [1080, 1361], [1041, 42], [876, 93], 'ty_cj_lsbx_ty', 0.9929825]]
    list_lsbx: [[[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456], 'ty_cj_lsbx', 0.9977392], [[726, 491], [694, 1372], [1014, 1372], [1079, 491], 'ty_cj_lsbx', 0.9945057]]
    list_logo: [[[398, 175], [398, 301], [995, 332], [995, 200], 'ty_cj_logo_ty', 0.9994031]]
    list_logo_fake: []
    list_cj: [[[278, 488], [278, 623], [497, 650], [655, 646], [655, 495], 'layer', 0.99953794], [[753, 506], [719, 607], [719, 820], [1008, 852], [1057, 663], [1057, 506], 'layer', 0.9751854], [[301, 1204], [301, 1319], [636, 1323], [636, 1203], 'layer', 0.9997247], [[716, 859], [716, 996], [730, 1007], [963, 1038], [1002, 1027], [1018, 961], [1018, 890], [816, 856], 'layer', 0.9949944], [[300, 1033], [300, 1169], [317, 1180], [638, 1180], [638, 1040], 'layer', 0.99958855], [[711, 1034], [711, 1182], [987, 1191], [997, 1180], [1009, 1075], [994, 1060], [823, 1028], [733, 1021], 'layer', 0.94295245], [[295, 847], [295, 985], [312, 1002], [635, 1002], [635, 867], 'layer', 0.9998399], [[702, 1206], [697, 1316], [713, 1325], [992, 1331], [992, 1214], 'layer', 0.9396977], [[288, 660], [288, 807], [304, 819], [627, 834], [638, 678], 'layer', 0.9998305]]
    '''

    num_ty_sh_lsbx = len(list_lsbx_ty) + len(list_mlsbx_sh)
    double_lsbx_temp = [[] for _ in range(num_ty_sh_lsbx)]
    double_lsbx = []
    soc_mlsbx_temp = [[] for _ in range(num_ty_sh_lsbx)]
    soc_mlsbx = []

    list_mlsbx_sh = check_by_logo(list_mlsbx_sh, list_logo_fake, 'ty_cj_lsbx_ty', 0.54)
    for cjsku in copy.deepcopy(list_mlsbx_sh):
        if cjsku[-2] == "ty_cj_lsbx_ty":
            list_cj.append(cjsku)
            list_mlsbx_sh.remove(cjsku)
    list_lsbx_ty = cj_nms(list_lsbx_ty)

    list_lsbx_del = []
    if len(list_lsbx):
        if len(list_logo_fake):
            for i in range(len(list_lsbx)):
                for fake in list_logo_fake:
                    list_lsbx[i] = get_box_center(fake, list_lsbx[i])
        all_in_ty = False
        for cjtemp in list_lsbx:
            in_ty, index_ty = juge_iou(cjtemp, list_lsbx_ty)
            in_sh, index_sh = juge_iou(cjtemp, list_mlsbx_sh)
            if in_ty:
                list_lsbx_del.append(cjtemp)
                cjtemp[-2] = 'ty_lsbx_cj'  # 将 'ty_cj_lsbx' -> 'ty_lsbx_cj'
                double_lsbx_temp[index_ty].append(cjtemp)  # 将tylsbx里的lsbx放入double_lsbx_temp
                all_in_ty = True
            elif in_sh and not in_ty:
                if cjtemp not in soc_mlsbx_temp[index_sh]:
                    soc_mlsbx_temp[index_sh].append(cjtemp)
            elif not in_ty and not in_sh:
                for fake in list_logo_fake:
                    cjtemp = get_box_center(fake, cjtemp)  # 加入对new_logo的处理
                list_cj.append(cjtemp)

        for del_lsbx in list_lsbx_del:
            try:  # 防止多删
                list_cj.remove(del_lsbx)
            except:
                pass

        if not all_in_ty:
            check_by_logo(list_lsbx_ty, list_logo, 'ty_lsbx_cj')
            list_cj.extend(list_lsbx_ty)

    # 处理只有lsbx_ty和logo的情况
    else:
        check_by_logo(list_lsbx_ty, list_logo, 'ty_lsbx_cj')
        list_cj.extend(list_lsbx_ty)

    if any(double_lsbx_temp):  # 有一个不为空
        for l in double_lsbx_temp:
            if len(l) >= 2:
                double_lsbx.append(l)
            elif len(l) == 1:
                list_cj.append(l[0])

    # list_cj=[[278, 488], [278, 623], [497, 650], [655, 646], [655, 495], 'layer', 0.99953794], [[753, 506], [719, 607], [719, 820], [1008, 852], [1057, 663], [1057, 506], 'layer', 0.9751854], [[301, 1204], [301, 1319], [636, 1323], [636, 1203], 'layer', 0.9997247]
    # double_lsbx=[[ [[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456], 'ty_lsbx_cj', 0.9977392], [[726, 491], [694, 1372], [1014, 1372], [1079, 491], 'ty_lsbx_cj', 0.9945057] ],]
    if any(soc_mlsbx_temp):
        for l in soc_mlsbx_temp:
            if len(l) >= 2:
                soc_mlsbx.append(l)
            elif len(l) == 1:
                list_cj.append(l[0])
    return list_cj, double_lsbx, soc_mlsbx


def check_by_logo(list_cjsku, list_logo, target, threshold=0.8):
    if len(list_logo) == 0:
        return list_cjsku
    for cjsku in list_cjsku:
        for logotemp in list_logo:
            iou = calculate_iou(logotemp[-1], cjsku[-1])
            if iou > threshold:
                cjsku[-2] = target
                break
    return list_cjsku


def juge_iou(cjsku, target_cj_list, threshold=0.8):
    in_cj = False
    loc_index = 0
    for index, targetcj in enumerate(target_cj_list):
        iou = calculate_iou(cjsku[-1], targetcj[-1])
        if iou > threshold:
            in_cj = True
            loc_index = index
            break
    return in_cj, loc_index


def get_box_center(box1, box2):
    # box1: fake;  box2: ty_cj_lsbx
    if box2[-2] == 'ty_lsbx_cj':
        return box2
    p = Polygon(box1[-1]).centroid
    flag = Polygon(box2[-1]).contains(p)
    if not flag:
        box2_xmin = min([i[0] for i in box2[-1]])
        box2_xmax = max([i[0] for i in box2[-1]])
        box2_ymin = min([i[1] for i in box2[-1]])
        box2_ymax = max([i[1] for i in box2[-1]])
        if 0 < (box2_xmax - box2_xmin) / (box2_ymax - box2_ymin) <= 1:  # 宽高比∈(0,1]，正图
            if box2_xmin <= p.x <= box2_xmax:
                flag = True
            else:  # 2.x的框缩小了导致fake中心点不在lsbx的框内
                box1_xmin = min([i[0] for i in box1[-1]])
                box1_xmax = max([i[0] for i in box1[-1]])
                if box2_xmin < box1_xmax < box2_xmax or box2_xmin < box1_xmin < box2_xmax:
                    flag = True
        else:  # 旋转图
            if box2_ymin <= p.y <= box2_ymax:
                flag = True
            else:  # 2.x的框缩小了导致fake中心点不在lsbx的框内
                box1_ymin = min([i[1] for i in box1[-1]])
                box1_ymax = max([i[1] for i in box1[-1]])
                if box2_ymin < box1_ymax < box2_ymax or box2_ymin < box1_ymin < box2_ymax:
                    flag = True
    if flag:
        box2[-2] = 'ty_lsbx_cj'
    return box2


def procces_ryj(list_ryj, list_ryj_logo, list_ryj_fake, list_logo, list_logo_fake, list_cj):
    """
    热饮机后处理逻辑
    update: 2021/4/13 by liuyu
    Args:
        list_logo_fake:新增参数，部分热饮机有ty_cj_logo_ty_fake
    Return:
        list_cj: 其他场景和处理后的热饮机场景合并结果
    """
    # list_ryj_logo_ty = []
    # for ryj_logo in list_ryj_logo:
    #     cj_check_and_juge(ryj_logo, list_ryj_fake, "ty_cj_ryj_logo", 0.8)
    #     list_ryj_logo_ty.append(ryj_logo)

    if len(list_ryj):  # 场景中有热饮机
        for ryj in list_ryj:
            cj_check_and_juge(ryj, list_logo, 'ty_ryj_cj', 0.8)
            # cj_check_and_juge(ryj, list_ryj_logo_ty, 'ty_ryj_cj', 0.8)
            cj_check_and_juge(ryj, list_ryj_fake, 'ty_ryj_cj', 0.8)
            cj_check_and_juge(ryj, list_logo_fake, 'ty_ryj_cj', 0.7)
            list_cj.append(ryj)
    return list_cj


def cj_check_and_juge(cjsku, list_logo, target_cj, iou_threshold):
    if len(list_logo) > 0 and cjsku[-2] != target_cj:  # 存在ty_logo
        for logotemp in list_logo:
            iou = calculate_iou(logotemp[-1], cjsku[-1])
            p = Polygon(logotemp[-1]).centroid
            flag = Polygon(cjsku[-1]).contains(p)
            if iou > iou_threshold or flag:
                cjsku[-2] = target_cj


def calculate_iou(polygon1, polygon2):
    """
    计算两个多边形的iou
    update: 2020/12/16 by linmuxing
    Args:
        polygon1/polygon2: mask轮廓点数组，[[x1,y1],[]]

    Return:
        iou: 两多边形的交并比

    """
    poly1 = Polygon(polygon1)  # Polygon：多边形对象
    poly1 = poly1.buffer(0.01)
    poly2 = Polygon(polygon2)
    poly2 = poly2.buffer(0.01)
    area1, area2 = poly1.area, poly2.area
    # print(poly1,poly2)
    min_area = min(area1, area2)
    try:
        if not poly1.intersects(poly2):
            inter_area = 0  # 如果两四边形不相交
        else:
            inter_area = poly1.intersection(poly2).area  # 相交面积
    except:
        inter_area = 0
    iou = inter_area / min_area
    # print(iou)
    return iou


def getSceneInfo(mask_dict):  # 统计场景类别、类别数量、场景列表
    ''' 基于实例分割的字典生成场景信息
    Args:
        mask_dict: 字典,实例分割的字典信息

    Returns:
        scene_info: 字典,场景信息
    '''
    # pdb.set_trace();print('get-Scene-Info')
    scene_info = {}
    scene_list = []
    scene_location = {}
    # shapes = mask_dict["shapes"] #原
    shapes = mask_dict["shapes"] + [j for i in mask_dict["double_lsbx"]["shapes"] for j in i] + \
             [jj for ii in mask_dict["soc_mlsbx"]["shapes"] for jj in ii]
    if (len(shapes) > 0):
        for ele in shapes:
            label = ele['label']
            scene_label = label.split('-')[0]
            if scene_label not in ['layer', 'ty_cj_lsbx_fake', 'ty_cj_lsbx_ty', 'ty_cj_logo_ty']:
                scene_list.append(scene_label)
                scene_location[label] = ele['points']

        # if mask_dict['double_lsbx']['shapes']: #没有旋转信息

        scene_list.append('ty_cj_others')  # <- 加入了others
        scene_set = set(scene_list)
        scene_location['ty_cj_others'] = []

        for scene_name in scene_set:
            scene_info.update({scene_name: 0})

        for scene_name in scene_list:
            scene_info[scene_name] += 1
        scene_info['sceneLocation'] = scene_location
        scene_info.update({'sceneSetNum': len(scene_set)})
        scene_info.update({'sceneSet': scene_set})
        scene_info.update({'sceneList': scene_list})

    # scene_info={'ty_cj_others': 1, 'ty_lsbx_cj': 2, 'sceneLocation': {'ty_lsbx_cj-1': [[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456]],
    #            'ty_lsbx_cj-2': [[726, 491], [694, 1372], [1014, 1372], [1079, 491]], 'ty_cj_others': []},
    #            'sceneSetNum': 2, 'sceneSet': {'ty_cj_others', 'ty_lsbx_cj'}, 'sceneList': ['ty_lsbx_cj', 'ty_lsbx_cj', 'ty_cj_others']}
    return scene_info


# def postProcess(skudata, imgInfo, thres, skuslist=None, scene=None):
def postProcess(skudata, imgInfo, scene=None):
    # step5: To put the skus into different layers using the stacked skulist
    # pdb.set_trace()
    cj_sku = {}
    new_skudata = {'boxInfo': []}
    img_h = imgInfo['height']
    img_w = imgInfo['width']
    if scene:

        sku_info, skudata['boxInfo'] = skuInstance(skudata['boxInfo'])

        # # 场景信息
        cj_info = scene
        skuClass = [ele['skuName'] for ele in sku_info]
        sku_coords = compute_all_centroid(sku_info, img_h, img_w)  # 计算所有sku的质心
        # sku 分配到各场景
        cj_sku = {}
        for cj, point in cj_info.items():
            if 'cj' in cj:
                temp = []
                for sku in skuClass:
                    if isInsidePolygon(sku_coords[sku], point):
                        temp.append(sku)
                cj_sku[cj] = temp

        # 把已分配的SKU从skulist和skudata中删除
        # _, skuslist = splitSkuList(skuslist, cj_sku)

        new_skudata = skudata.copy()
        new_skudata['boxInfo'], skudata['boxInfo'] = splitSkuList(skudata['boxInfo'], cj_sku)
        skus = new_skudata['boxInfo']
        new_skudata['skuStat'] = skuStatus(skus)

    for box in skudata['boxInfo']:
        box['skuName'] = box['skuName'].split('-')[0]
    skus = skudata['boxInfo']
    skudata['skuStat'] = skuStatus(skus)

    return new_skudata, skudata, cj_sku


def compute_all_centroid(all_info, img_h, img_w):
    ''' 计算所有SKU的中心点坐标
    Args:
       all_info: 所有sku的信息
       img_h, img_w: 图像的高度和宽度
    Returns:
       centroid_dict: 字典, {'skuname-i':[x1, y1], 'skuname-i':[x1, y1]}
    '''
    centroid_dict = {}
    for ele in all_info:
        point = point_trans(ele, img_w, img_h)
        # point = ele['points']
        centroid = compute_centroid(point)
        centroid_dict[ele['skuName']] = centroid
    return centroid_dict


def point_trans(ele, img_w, img_h):
    point = [
        [ele['location']['xmin'] * img_w, ele['location']['ymin'] * img_h],
        [ele['location']['xmax'] * img_w, ele['location']['ymin'] * img_h],
        [ele['location']['xmax'] * img_w, ele['location']['ymax'] * img_h],
        [ele['location']['xmin'] * img_w, ele['location']['ymax'] * img_h]
    ]
    return point


def compute_centroid(points):
    ''' 计算多边形的中心点坐标
    Args:
        points: 轮廓点

    Returns:
        [mx, my] 多边形的中心点坐标
    '''
    temp = Polygon(points)
    m_x = (float(temp.centroid.wkt.split(' ')[1][1:]))
    m_y = (float(temp.centroid.wkt.split(' ')[2][:-1]))
    return [m_x, m_y]


def fomart_data_without_cj1(res):  # 应主要处理图中完全没有识别到场景的情况
    imgW, imgH = res['imageInfo']['width'], res['imageInfo']['height']
    if 'layerNum' not in res:
        return [{'placeType': 0, 'placeSegname': 'ty_cj_others', 'area': 0, 'boxInfo': res['boxInfo'],
                 'placeLocation': [[0, 0], [0, imgH], [imgW, imgH], [imgW, 0]], 'skuStat': res['skuStat'],
                 'layerOut': [], 'layerNum': 0, 'layerLine': []}]
    result = []
    data = {'placeType': 0, 'placeSegname': 'ty_cj_others', 'area': 0, 'boxInfo': res['boxInfo'],
            'placeLocation': [[0, 0], [0, imgH], [imgW, imgH], [imgW, 0]],
            'skuStat': res['skuStat'], 'layerOut': res['layerOut'], 'layerNum': res['layerNum'],
            'layerLine': res['layerLine']}
    result.append(data)
    # return [res]  # 关闭函数
    return result


def getCjinfo(mask_dict, sku):
    # scene_list = []
    scene_location = {}
    shapes = mask_dict["shapes"]
    shapes_cp = shapes.copy()
    if (len(shapes) > 0):
        for ele in shapes:
            label = ele['label']
            if label.split('-')[0] in sku:
                scene_location[label] = ele['points']
                shapes_cp.remove(ele)
        mask_dict["shapes"] = shapes_cp
    return scene_location, mask_dict


def skuInfoFormat(res):
    ''' 对SKU重新计算 state
    '''
    if res:
        skus = res['boxInfo']
        skuStat = skuStatus(skus)
        res['skuStat'] = skuStat
    return res


def fomart_data(res, scene_info, cj_sku=None, res_cj=None):
    # pdb.set_trace();print('【format_data】')
    result = []
    if res_cj:  # lsbx类
        # ly 增加双门tylsbx融合
        tylsbx_gates = {}
        if 'dlsbxMerge' in res_cj.keys() and res_cj['dlsbxMerge']:
            minus_count = 0  # 被合并入多开门冰箱的lsbx
            for lsbxcj, lsbxcjlist in res_cj['dlsbxMerge'].items():
                merge_list = []
                minus_count += (len(lsbxcjlist) - 1)
                for cj in lsbxcjlist:
                    merge_list.append(scene_info['sceneLocation'][cj])
                    del scene_info['sceneLocation'][cj]

                if res_cj['ifRot']:  # xuanzhuan
                    merge_location = merge_lsbx(merge_list, ifRot=True)
                    scene_info['sceneLocation'][lsbxcj] = merge_location
                else:  # zhengchang
                    merge_location = merge_lsbx(merge_list, ifRot=False)
                    scene_info['sceneLocation'][lsbxcj] = merge_location

                tylsbx_gates[lsbxcj] = len(lsbxcjlist)  # 记录这一场景的门数
            scene_info['ty_lsbx_cj'] -= minus_count
            for k in range(minus_count):
                scene_info['sceneList'].remove('ty_lsbx_cj')

        # ly 增加多门社会lsbx融合
        # pdb.set_trace();print('==shlsbx==')
        shlsbx_gates = {}
        if 'shlsbxMerge' in res_cj.keys() and res_cj['shlsbxMerge']:
            minus_count = 0  # 被合并入多开门冰箱的lsbx
            for lsbxcj, lsbxcjlist in res_cj['shlsbxMerge'].items():
                merge_list = []
                minus_count += (len(lsbxcjlist) - 1)
                for cj in lsbxcjlist:
                    merge_list.append(scene_info['sceneLocation'][cj])
                    del scene_info['sceneLocation'][cj]

                if res_cj['ifRot']:  # xuanzhuan
                    merge_location = merge_lsbx(merge_list, ifRot=True)
                    scene_info['sceneLocation'][lsbxcj] = merge_location
                else:  # zhengchang
                    merge_location = merge_lsbx(merge_list, ifRot=False)
                    scene_info['sceneLocation'][lsbxcj] = merge_location

                shlsbx_gates[lsbxcj] = len(lsbxcjlist)  # 记录这一场景的门数
            scene_info['ty_cj_lsbx'] -= minus_count
            for k in range(minus_count):
                scene_info['sceneList'].remove('ty_cj_lsbx')

        for cj in res_cj['layerCount'].keys():
            if cj == 'ty_cj_others':  # <- new add
                continue  # <- new add
            # elif layerNum == 0: # 加入layerNum==0主要删除小面积lsbx无层
            #     continue
            data = {}
            data['placeType'] = cj_map[cj.split('-')[0]]
            data['placeSegname'] = cj
            data['area'] = 0
            if 'boxInfo' in res.keys():
                boxInfo = [res['boxInfo'][i] for i in range(len(res['boxInfo'])) if res['boxInfo'][i]['skuName'] in
                           res_cj['skuCj'] and res_cj['skuCj'][res['boxInfo'][i]['skuName']] == cj]
            else:
                boxInfo = []
            # print(res_cj['skuCj'][res['boxInfo'][i]['skuName']])
            data['boxInfo'] = boxInfo
            data['layerNum'] = res_cj['layerCount'][cj]
            data['layerOut'] = res['layerOut'] if 'layerOut' in res.keys() else []
            data['layerLine'] = res['layerLine'] if 'layerLine' in res.keys() else []
            data['skuStat'] = skuStatus(boxInfo)
            if cj in tylsbx_gates.keys():
                data['specification'] = tylsbx_gates[cj]
            elif cj in shlsbx_gates.keys():
                data['specification'] = shlsbx_gates[cj]
            elif cj.split('-')[0] in ['ty_lsbx_cj', 'ty_cj_lsbx']:
                data['specification'] = 1
            else:
                data['specification'] = 0
            # print('specification:', data['specification'])
            result.append(data)

        for d in result:
            if d['placeSegname'] in scene_info['sceneLocation'].keys():
                d['placeLocation'] = scene_info['sceneLocation'][d['placeSegname']]

    elif cj_sku:
        for cj, sku in cj_sku.items():
            if cj == 'ty_cj_others':  # <- new add
                continue  # <- new add
            data = {}
            data['placeType'] = cj_map[cj.split('-')[0]]
            data['placeSegname'] = cj
            data['area'] = 0
            boxInfo = [res['boxInfo'][i] for i in range(len(res['boxInfo'])) if res['boxInfo'][i]['skuName'] in sku]
            data['boxInfo'] = boxInfo
            data['layerNum'] = max([int(box['layer'].strip() or 0) for box in res['boxInfo']]) if res[
                'boxInfo'] else 0  # 这里原先代码有问题,转int
            data['layerOut'] = res['layerOut'] if 'layerOut' in res.keys() else []
            data['layerLine'] = res['layerLine'] if 'layerLine' in res.keys() else []
            data['skuStat'] = skuStatus(boxInfo)
            data['specification'] = 0
            # print('specification:', data['specification'])
            result.append(data)
        for d in result:
            d['placeLocation'] = scene_info['sceneLocation'][d['placeSegname']]
    return result


def merge_lsbx(merge_list, ifRot=False):
    # len(merge_list) > 1 一定
    # print('merge门数：', len(merge_list))
    x1_coords = [i[0] for i in merge_list[0]]
    y1_coords = [i[1] for i in merge_list[0]]
    x2_coords = [i[0] for i in merge_list[-1]]
    y2_coords = [i[1] for i in merge_list[-1]]
    tl1, bl1, tr1, br1 = [min(x1_coords), min(y1_coords)], [min(x1_coords), max(y1_coords)], \
                         [max(x1_coords), min(y1_coords)], [max(x1_coords), max(y1_coords)]
    tl2, bl2, tr2, br2 = [min(x2_coords), min(y2_coords)], [min(x2_coords), max(y2_coords)], \
                         [max(x2_coords), min(y2_coords)], [max(x2_coords), max(y2_coords)]
    if ifRot:
        return [tl1, tr1, br1, tr2, br2, bl2, tl2, bl1]
    else:  # zhengchang
        if (tr1[0] - tl1[0]) > (bl1[1] - tl1[1]) or (tr2[0] - tl2[0]) > (bl2[1] - tl2[1]):  # 旋转的图但没检出旋转
            if (bl1[1] + tl1[1]) < (bl2[1] + tl2[1]):
                return [tl1, tr1, br1, tr2, br2, bl2, tl2, bl1]
            else:
                return [tl2, tr2, br2, tr1, br1, bl1, tl1, bl2]
        else:
            return [tl1, tr1, tl2, tr2, br2, bl2, br1, bl1]  # 原本正的图


def layerDetectMask(skudata, layerdata, imgInfo):
    '''基于SKU信息以及场景分层实例信息进行分层判断

    Args:
       skudata: 字典, 存放的是SKU识别信息
       layerdata: 字典, 存放的是场景分层的信息
       imgInfo:


    Returns:
       res: 字典,返回后的json数据
       res_fc: 字典,分层信息

    '''
    # pdb.set_trace()
    res = {}
    res_fc = {}
    cjlist = [ele['label'] for ele in layerdata['shapes'] if
              ele['label'].split('-')[0] not in ['layer', 'ty_cj_lsbx_fake', 'ty_cj_lsbx_ty',
                                                 'ty_cj_logo_ty']]  # cjlist=['ty_lsbx_cj-1', 'ty_lsbx_cj-2']
    cjlist_d_lsbx = [ele['label'] for ele in [j for i in layerdata['double_lsbx']['shapes'] for j in i]
                     if ele['label'].split('-')[0] not in ['layer', 'ty_cj_lsbx_fake', 'ty_cj_lsbx_ty',
                                                           'ty_cj_logo_ty']]  # 增加这句，提取double_lsbx的场景信息放入cjlist
    cjlist_sh_lsbx = [ele['label'] for ele in [j for i in layerdata['soc_mlsbx']['shapes'] for j in i]
                      if ele['label'].split('-')[0] not in ['layer', 'ty_cj_lsbx_fake', 'ty_cj_lsbx_ty',
                                                            'ty_cj_logo_ty']]  # 增加这句，提取double_lsbx的场景信息放入cjlist
    cjlist += (cjlist_d_lsbx + cjlist_sh_lsbx)  # cjlist=['ty_cj_lsbx-1', 'ty_cj_lsbx-1', 'ty_cj_lsbx-2']
    if cjlist and skudata['boxInfo']:
        res = skudata
        res_fc = layer_detect(imgInfo, skudata, layerdata)  # layer_detect 计算层排序等
        # pdb.set_trace()
        for i in range(len(res['boxInfo'])):
            if 'layerOut' in res['boxInfo'][i]:
                del res['boxInfo'][i]['layerOut']

        # 计算目标SKU的场景用于确定主场景
        sku_scene = []  # 目标sku的场景
        count_scene = {}

        for i in range(len(res['boxInfo'])):
            res['boxInfo'][i]['layer'] = res_fc['layer'][res['boxInfo'][i]['skuName']]  # 修改层的编号为真实层数？
            if res['boxInfo'][i]['layer'] != 0:
                # if not res['boxInfo'][i]['skuName'].startswith('PPPPPPP_add'):# 这里有必要过滤掉PPPPPPP_add？
                #     sku_scene.append(res_fc['skuCj'][res['boxInfo'][i]['skuName']]) #放入含有sku的场景
                if res['boxInfo'][i]['skuName'] in res_fc['skuCj']:
                    sku_scene.append(res_fc['skuCj'][res['boxInfo'][i]['skuName']])

        # pdb.set_trace();print('!!!!')
        main_scene = list(set(sku_scene))  # 合并后的lsbx名
        for ele in main_scene:
            count_scene[ele] = sku_scene.count(ele)  # {'ty_lsbx_cj-1': 9, 'ty_lsbx_cj-2': 4} #场景中的sku数量 不包含PPPPPPP_add

        if len(main_scene) == 1:  # 如果场景数为1, 那么就把该场景的层数赋值给res['layerNum']
            res['layerNum'] = res_fc['layerCount'][main_scene[0]]
        elif len(main_scene) > 1:  # 如果场景数大于1, 那么就把SKU最多的那个场景的层数赋值给res['layerNum']
            state = 0
            sorted_list = sorted(count_scene.items(), key=lambda x: x[1],
                                 reverse=True)  # [('ty_cj_lsbx-2', 50), ('ty_cj_lsbx-1', 41)]
            for scene in sorted_list:
                if scene[0] in res_fc['layerCount'].keys() and not state:
                    res['layerNum'] = res_fc['layerCount'][scene[0]]
                    state = 1
            if not state:
                res['layerNum'] = 0
            # temp_scene = max(count_scene, key = count_scene.get)  #不使用直接计算最大值的方法，以避免value相等时key不确定的问题。ly0907。
            # res['layerNum'] = res_fc['layerCount'][temp_scene]
        else:  # 如果场景数为0, 那么就把layerCount的内容赋值给res['layerNum']
            # res['layerNum'] = max(res_fc['layerCount'].values())
            res['layerNum'] = 0

            # pdb.set_trace()
        res['layerLine'] = []
        res['layerOut'] = res_fc['layerOut']

        # 删掉冰柜场景中没有在场景框内的SKU，但是没有删掉在场景框内不在层内的SKU
        # 目的是为了解决冰柜门上出现的倒影SKU等...
        res = rmZeroLayer(res, main_scene)
    elif cjlist and not skudata['boxInfo']:
        res_fc['layerCount'] = {cj: 0 for cj in cjlist}  # res_fc={'layerCount': {'ty_cj_hj-1': 0}};res={}
    return res, res_fc


def layer_detect(imgInfo, skudata, layerdata):
    '''采用mask对场景内的SKU进行分层
    Args:
       img: 数组, opencv读入的图像数组
       skudata: 字典, 识别出的SKU信息
       layerdata: 字典, 场景分层的信息

    Returns:
    res['layer'] = res_fc
    res['skuCj'] = sku_cj
    res['layerCount'] = layer_count
    res['layerOut'] = finalLayerout

    res_fc = {'yl_ty_dgz_xc310P_fake-4': '3', 'yl_ty_hzy_hjl_fake-4': '3', 'yl_ty_jjnm_500P_fake-1': '5', 'yl_ty_asm_yyzsnc450P_fake-1': '5', 'yl_ty_asm_yyzsnc450P_fake-2': '5', 'yl_ty_asm_yyzsnc450P_fake-3': '5', 'yl_ty_asm_yyzsnc450P_fake-4': '5', 'yl_ty_bhc_310P_fake-1': '4', 'yl_ty_bhc_310P_fake-2': '4', 'yl_ty_bhc_310P_fake-3': '4', 'yl_ty_bhc_310P_fake-4': '4', '0000449-1': '4', '0000449-2': '4', '0000449-3': '4', '0003466-1': '2'} 每个SKU位于第几层

    sku_cj = {'yl_ty_hzy_hjl_fake-1': 'cj_lsbx-1', 'yl_ty_hzy_hjl_fake-2': 'cj_lsbx-1', 'yl_ty_hzy_hjl_fake-3': 'cj_lsbx-1', 'yl_ty_dgz_xc310P_fake-1': 'cj_lsbx-1', 'yl_ty_dgz_xc310P_fake-2': 'cj_lsbx-1', 'yl_ty_dgz_xc310P_fake-3': 'cj_lsbx-1', 'yl_ty_dgz_xc310P_fake-4': 'cj_lsbx-1', 'yl_ty_hzy_hjl_fake-4': 'cj_lsbx-1', 'yl_ty_jjnm_500P_fake-1': 'cj_lsbx-1', 'yl_ty_asm_yyzsnc450P_fake-1': 'cj_lsbx-1', 'yl_ty_asm_yyzsnc450P_fake-2': 'cj_lsbx-1', 'yl_ty_asm_yyzsnc450P_fake-3': 'cj_lsbx-1', 'yl_ty_asm_yyzsnc450P_fake-4': 'cj_lsbx-1', 'yl_ty_bhc_310P_fake-1': 'cj_lsbx-1', 'yl_ty_bhc_310P_fake-2': 'cj_lsbx-1', 'yl_ty_bhc_310P_fake-3': 'cj_lsbx-1', 'yl_ty_bhc_310P_fake-4': 'cj_lsbx-1', '0000449-1': 'cj_lsbx-1', '0000449-2': 'cj_lsbx-1', '0000449-3': 'cj_lsbx-1', '0003466-1': 'cj_lsbx-1', '0003466-2': 'cj_lsbx-1', 'yl_ty_hzy_nm_fake-1': 'cj_lsbx-1', 'yl_ty_hzy_nm_fake-2': 'cj_lsbx-1', 'yl_ty_xmtx_qnhc_fake-1': 'cj_lsbx-1', 'yl_ty_xmtx_qnhc_fake-2': 'cj_lsbx-1', 'yl_ty_asm_yw1500P_fake-1': 'cj_lsbx-1', 'yl_ty_asm_yw1500P_fake-2': 'cj_lsbx-1', 'yl_ty_xmtx_lldc_fake-1': 'cj_lsbx-1', 'yl_ty_xmtx_lldc_fake-2': 'cj_lsbx-1', 'yl_ty_xmtx_lldc_fake-3': 'cj_lsbx-1', 'yl_ty_xmtx_lldc_fake-4': 'cj_lsbx-1', 'yl_ty_xmtx_cjwl_fake-1': 'cj_lsbx-1', 'yl_ty_xmtx_cjwl_fake-2': 'cj_lsbx-1', 'yl_ty_xmtx_cjwl_fake-3': 'cj_lsbx-1'} 每个SKU位于哪个场景 但是缺少了没有在场景内的SKU

    layer_count = {'cj_lsbx-1': 5} 每个场景具有的层数

    finalLayerout = [{'skuName': '0000699', 'count': 1}, {'skuName': '0003435', 'count': 2}, {'skuName': '0000698', 'count': 1}, {'skuName': '0003026', 'count': 2}, {'skuName': '0003436', 'count': 2}, {'skuName': '0000004', 'count': 3}, {'skuName': '0001595', 'count': 1}, {'skuName': '0000700', 'count': 1}, {'skuName': 'QQQQQQQ', 'count': 1}, {'skuName': '0002282', 'count': 2}, {'skuName': '0003404', 'count': 2}, {'skuName': '0002283', 'count': 2}, {'skuName': '0003437', 'count': 4}]

    '''
    # pdb.set_trace();print('【layer_detect】')
    ##clgjSkuList = ['0002282', '0002283', '0003026', '0003041', '0003200', '0003257','0003404', '0003534', '0003535', '0003689', '0003729', '0003730', '0003731', '0003741', '0003880', '0003927', '0003928', '0003929', '0000004', '0003879', 'QQQQQQQ']
    # clgjSkuList = ['0002282', '0002283', '0003026', '0003041', '0003200', '0003257','0003404', '0003534', '0003535', '0003689', '0003729', '0003730', '0003731', '0003741', '0003880', '0003927', '0003928', '0003929', '0000004', '0003879']
    scene_info = getSceneInfo(
        layerdata)  # {'ty_lsbx_cj': 2, 'sceneSetNum': 1, 'sceneSet': {'ty_lsbx_cj'}, 'sceneList': ['ty_lsbx_cj', 'ty_lsbx_cj']}

    # sku信息,近似于把语义分割的SKU信息转换为实例分割的SKU信息
    sku_class = []  # sku名，再转集合
    for ele in skudata['boxInfo']:
        sku_name = ele['skuName']
        sku_class.append(sku_name)
    sku_class = set(sku_class)  # {'0003303', '0000433', '0003170'}{'PPPPPPP', '0001787', 'FFFFFFF', 'ty_xz_daiding'}

    sku_label_count = {}
    for sku_label in sku_class:
        sku_label_count.update({sku_label: 0})  # {'0003303': 0, '0000433': 0, '0003170': 0}
    sku_info = []
    for ele in skudata['boxInfo']:
        ele_info = ele  # ele:  {'skuName': 'daiding_101', 'location': {'xmin': 0.219, 'ymin': 0.584, 'xmax': 0.379, 'ymax': 0.652}, 'score': 96.0, 'layerOut': [3]}
        sku_label = ele['skuName']
        sku_label_count[sku_label] += 1
        ele_info['skuName'] = sku_label + '-' + str(sku_label_count[sku_label])
        sku_info.append(ele_info)
    # 得到sku_info=[{'layer': '1', 'skuName': '0003170-1', 'location': {'xmin': 0.281, 'ymin': 0.444, 'xmax': 0.327, 'ymax': 0.521}, 'score': 100.0}, {'layer': '1', 'skuName': '0003303-1', 'location': {'xmin': 0.368, 'ymin': 0.448, 'xmax': 0.413, 'ymax': 0.519}, 'score': 99.0}, {'layer': '2', 'skuName': '0000433-1', 'location': {'xmin': 0.301, 'ymin': 0.543, 'xmax': 0.357, 'ymax': 0.608}, 'score': 89.0}, {'layer': '2', 'skuName': '0000433-2', 'location': {'xmin': 0.356, 'ymin': 0.543, 'xmax': 0.41, 'ymax': 0.608}, 'score': 79.0}, ]

    # 场景和分层信息
    cjfc_info = layerdata['shapes']  # 场景+分层
    cjfc_dlsbx_info = layerdata['double_lsbx']['shapes']  # [ [{},{}], [{},{}] ]
    cjfc_shlsbx_info = layerdata['soc_mlsbx']['shapes']  # [ [{},{}], [{},{}] ]
    img_h = imgInfo['height']
    img_w = imgInfo['width']

    # 计算场景框和分层框的质心
    cj_info = []  # [{'line_color': None, 'fill_color': None, 'label': 'ty_lsbx_cj-1', 'shape_type': 'polygon', 'flags': {}, 'points': [[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456]]}, {'line_color': None, 'fill_color': None, 'label': 'ty_lsbx_cj-2', 'shape_type': 'polygon', 'flags': {}, 'points': [[726, 491], [694, 1372], [1014, 1372], [1079, 491]]}]
    layer_info = []  # [{'line_color': None, 'fill_color': None, 'label': 'layer-1', 'shape_type': 'polygon', 'flags': {}, 'points': [[278, 488], [278, 623], [497, 650], [655, 646], [655, 495]]}, {'line_color': None, 'fill_color': None, 'label': 'layer-2', 'shape_type': 'polygon', 'flags': {}, 'points': [[753, 506], [719, 607], [719, 820], [1008, 852], [1057, 663], [1057, 506]]}]
    cj_dlsbx_info = []  # ly 2021/6/9
    cj_shlsbx_info = []  # ly 2021/8/19
    cj_class = ['ty_cj_hj', 'ty_cj_clj', 'ty_cj_lsbx', 'ty_cj_tg', 'ty_cj_clgj', 'ty_lsbx_cj', 'ty_ryj_cj', 'ty_cj_ryj',
                'ty_cj_lfg']  # 后续应加入陈列挂架的分层信息 |割箱、卧式冰柜、地堆 不分层

    for ele in cjfc_info:  # 分离场景、layer
        if 'cj' in ele['label'] and (ele['label']).split('-')[0] in cj_class:
            cj_info.append(ele)  # 【cj_info】
        elif 'layer' in ele['label']:
            layer_info.append(ele)  # 【layer_info】
            # layer_points[ele['label']] = ele['points'] #ly

    for ind, ele_d in enumerate(cjfc_dlsbx_info):  # 取出双开门tylsbx
        cj_dlsbx_info.append([])
        for ele in ele_d:
            if 'cj' in ele['label'] and (ele['label']).split('-')[0] in cj_class:
                cj_dlsbx_info[ind].append(ele)

    for ind, ele_d in enumerate(cjfc_shlsbx_info):  # 取出双开门shlsbx
        cj_shlsbx_info.append([])
        for ele in ele_d:
            if 'cj' in ele['label'] and (ele['label']).split('-')[0] in cj_class:
                cj_shlsbx_info[ind].append(
                    ele)  # [[{'line_color': None, 'fill_color': None, 'label': 'ty_cj_lsbx-1', 'shape_type': 'polygon', 'flags': {}, 'points': [[24, 504], [23, 1299], [254, 1294], [206, 505]]},
                # {'line_color': None, 'fill_color': None, 'label': 'ty_cj_lsbx-2', 'shape_type': 'polygon', 'flags': {}, 'points': [[313, 524], [331, 1253], [539, 1275], [546, 535]]}]]
    layer_coords = mask_compute_all_centroid(layer_info, img_h,
                                             img_w)  # 计算所有分层框的质心 {'layer-1': [470.3179213436729, 566.6232782125279], 'layer-2': [887.0545199486049, 669.3681570914307], 'layer-3': [469.6879432624114, 1262.255319148936], 'layer-4': [864.6291173022004, 944.7176460432896], 'layer-5': [467.9388308768595, 1108.104246067404]}
    layer_ratios = mask_compute_ratio(layer_info, img_h,
                                      img_w)  # 计算所有分层框的长宽 {'layer-1': [162, 377], 'layer-2': [346, 338], 'layer-3': [120, 335], 'layer-4': [182, 302], 'layer-5': [147, 338], 'layer-6': [170, 298], 'layer-7': [155, 340], 'layer-8': [125, 295], 'layer-9': [174, 350]}
    sku_coords = compute_all_centroid(sku_info, img_h,
                                      img_w)  # 计算所有sku的质心 {'0003170-1': [328.624, 926.3999999999999], '0003170-2': [375.107, 928.32], '0003303-1': [422.1305, 928.3199999999998], '0000433-1': [355.649, 1104.96], '0000433-2': [414.023, 1104.96], '0000433-3': [455.101, 1104.0]}

    # 获得所有SKU，层，场景的类别名 【skuClass】【layerClass】【cjClass】【cjDlsbxClass】【cjShlsbxClass】
    skuClass = []
    for ele in sku_info:
        skuClass.append(ele['skuName'])  # ['daiding_101-1', 'daiding_101-2', 'daiding_101-3', '0003730-1']
    layerClass = []
    for ele in layer_info:
        layerClass.append(ele[
                              'label'])  # ['layer-1', 'layer-2', 'layer-3', 'layer-4', 'layer-5', 'layer-6', 'layer-7', 'layer-8', 'layer-9']
    cjClass = []
    for ele in cj_info:
        cjClass.append(ele['label'])  # ['ty_cj_clj-1', 'ty_cj_hj-1']

    cjDlsbxClass = []
    cj_dlsbx_INFO = [j for i in cj_dlsbx_info for j in i]
    for ele in cj_dlsbx_INFO:
        cjDlsbxClass.append(ele['label'])  # cjDlsbxClass=['ty_lsbx_cj-1', 'ty_lsbx_cj-2']

    cjShlsbxClass = []
    cj_shlsbx_INFO = [j for i in cj_shlsbx_info for j in i]
    for ele in cj_shlsbx_INFO:
        cjShlsbxClass.append(ele['label'])  # cjDlsbxClass=['ty_lsbx_cj-1', 'ty_lsbx_cj-2']

    #############################################
    #       All informations are ready.         #
    #############################################
    # pdb.set_trace();print('cj-layer')
    # 合并场景、分层框、sku 缺少了 cj_sku
    cj_layer = {}  # 【场景-层】 【重要变量1】
    cj_layer_dlsbx = []  # 这里建一个列表存放双门lsbx  [{'ty_cj_clj-1': ['layer-5', 'layer-6'], 'ty_cj_hj-1': ['layer-1', 'layer-2']}, {'ty_cj_clj-3': ['layer-3', 'layer-4'], 'ty_cj_hj-4': ['layer-9', 'layer-8']}]
    cj_layer_shlsbx = []
    ratio_count = 0

    layers_add_PPP = []  # 特定场景需添加QQQQQQQ的层
    # 把层名分配给每个场景
    # pdb.set_trace()
    for i in range(len(cj_info)):  # 【场景】
        cj = cj_info[i]['label']  # 'ty_lsbx_cj-1'
        cj_point = cj_info[i]['points']  # [[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456]]
        temp = []
        for layer in layerClass:  # 【层】
            if isInsidePolygon(layer_coords[layer], cj_point):  # 【layer质心是否在场景框中】
                temp.append(layer)
                if cj.split('-')[0] in ['ty_lsbx_cj', 'ty_cj_lsbx', 'ty_cj_ryj', 'ty_ryj_cj']:  # 这些场景加待定bottle
                    layers_add_PPP.append(layer)
                if layer_ratios[layer][0] > 0.6 * img_h and layer_ratios[layer][1] < 0.5 * img_w or layer_ratios[layer][
                    0] > 0.7 * img_h:  # 高大于图高的0.6
                    ratio_count += 1  # 用于判断图像是否已经旋转

        cj_layer[
            cj] = temp  # 每个场景具有的层信息  {'ty_cj_clj-1': ['layer-5', 'layer-6', 'layer-9'], 'ty_cj_hj-1': ['layer-1', 'layer-2', 'layer-3', 'layer-4', 'layer-7', 'layer-8']}
        # {'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9'], 'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-6', 'layer-8']}  #这里包括了不含sku的空层

    # pdb.set_trace()
    for ind, ele_d in enumerate(
            cj_dlsbx_info):  # [[{'line_color': None, 'fill_color': None, 'label': 'ty_lsbx_cj-1', 'shape_type': 'polygon', 'flags': {}, 'points': [[262, 456], [291, 1337], [650, 1337], [674, 503], [638, 456]]}, {'line_color': None, 'fill_color': None, 'label': 'ty_lsbx_cj-2', 'shape_type': 'polygon', 'flags': {}, 'points': [[726, 491], [694, 1372], [1014, 1372], [1079, 491]]}]]
        temp = {}
        for i in range(len(ele_d)):
            temp_layer = []
            cj = ele_d[i]['label']
            cj_point = ele_d[i]['points']
            for layer in layerClass:
                if isInsidePolygon(layer_coords[layer], cj_point):  # layer质心是否在场景框中
                    temp_layer.append(layer)
                    layers_add_PPP.append(layer)
                    if layer_ratios[layer][0] > 0.6 * img_h and layer_ratios[layer][1] < 0.5 * img_w or \
                            layer_ratios[layer][0] > 0.7 * img_h:  # 高大于图高的0.6
                        ratio_count += 1  # 用于判断图像是否已经旋转

            temp[cj] = temp_layer
        cj_layer_dlsbx.append(temp)  # [ {'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9'],
        #   'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-6', 'layer-8']} ]
    flat_cj_layer_dlsbx = {}
    for ele in cj_layer_dlsbx:
        flat_cj_layer_dlsbx.update(ele)

    # pdb.set_trace()
    for ind, ele_d in enumerate(cj_shlsbx_info):
        temp = {}
        for i in range(len(ele_d)):
            temp_layer = []
            cj = ele_d[i]['label']
            cj_point = ele_d[i]['points']
            for layer in layerClass:
                if isInsidePolygon(layer_coords[layer], cj_point):  # layer质心是否在场景框中
                    temp_layer.append(layer)
                    layers_add_PPP.append(layer)
                    if layer_ratios[layer][0] > 0.6 * img_h and layer_ratios[layer][1] < 0.5 * img_w or \
                            layer_ratios[layer][0] > 0.7 * img_h:  # 高大于图高的0.6
                        ratio_count += 1  # 用于判断图像是否已经旋转

            temp[cj] = temp_layer
        cj_layer_shlsbx.append(temp)  # [ {'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9'],
        #   'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-6', 'layer-8']} ]
    flat_cj_layer_shlsbx = {}
    for ele in cj_layer_shlsbx:
        flat_cj_layer_shlsbx.update(ele)

    # pdb.set_trace();print('layer-sku')
    #################################################################################################################
    '''    
    # 把SKU名分配每个层 modify by lmx
    layer_sku = {}
    exact_sku = {}
    for j in range(len(layer_info)):  #【层】
        # pdb.set_trace()
        layer = layer_info[j]['label']    #'layer-1'
        layer_point = layer_info[j]['points'] #[[278, 488], [278, 623], [497, 650], [655, 646], [655, 495]]
        temp = []
        for sku in skuClass:         #【sku】
            if isInsidePolygon(sku_coords[sku], layer_point):  #如果sku中心点在层中
                temp.append(sku)
            else:          #这里直接分配给最近的层 逻辑有些问题 可能导致最终结果少部分sku
                for i in range(len(cj_info)):  #【场景】
                    cj = cj_info[i]['label']
                    cj_point = cj_info[i]['points']
                    if isInsidePolygon(sku_coords[sku], cj_point): #如果sku中心点在场景中
                        dis = {layer_info[j]['label']: pdist([layer_coords[layer_info[j]['label']], sku_coords[sku]]) for j in range(len(layer_info))}
                        # dis1 = {layer: pdist([layer_coords[layer], sku_coords[sku]]) for j in range(len(layer_info))}
                        sdis = sorted(dis.items(), key = lambda kv:(kv[1], kv[0]))
                        # print(sdis)
                        if sdis[0][0] in exact_sku.keys():
                            exact_sku[sdis[0][0]].append(sku)
                        else:
                            exact_sku[sdis[0][0]] = [sku]
        # 在layer中的sku：
        layer_sku[layer] = temp #layer_sku 每个层具有的sku信息 {'layer-1': [], 'layer-2': [], 'layer-3': [], 'layer-4': [], 'layer-5': ['daiding_101-2'], 'layer-6': ['daiding_101-1', 'daiding_101-3'], 'layer-7': ['0003730-1'], 'layer-8': [], 'layer-9': []}
    for lay, sku in exact_sku.items():
        if lay in layer_sku.keys():
            layer_sku[lay] = layer_sku[lay] + sku   # {'layer-1': ['PPPPPPP-3', 'PPPPPPP-5', 'FFFFFFF-1']}
    '''
    ###################################################################################################################
    # 把SKU名分配每个层 modify by ly 2021/07/01
    layer_sku = {}  # 【层-SKU】【重要变量2】
    exact_sku = {}
    in_layer_sku = []
    for j in range(len(layer_info)):  # 【层】
        # pdb.set_trace()
        layer = layer_info[j]['label']  # 'layer-1'
        layer_point = layer_info[j]['points']  # [[278, 488], [278, 623], [497, 650], [655, 646], [655, 495]]
        temp = []
        for sku in skuClass:  # 【sku】
            if isInsidePolygon(sku_coords[sku], layer_point):  # 如果sku中心点在层中
                temp.append(sku)
                in_layer_sku.append(sku)
        # 在layer中的sku：
        layer_sku[layer] = temp
    # pdb.set_trace()
    outLayerSku_cj = {}  # 分配不在层中的sku
    for sku in skuClass:  # 【SKU】
        if sku in in_layer_sku:  # 只看不在层中的sku
            continue
        in_common_cj = False  # 是否在非双开门lsbx中
        for i in range(len(cj_info)):  # 【场景】
            cj = cj_info[i]['label']
            cj_point = cj_info[i]['points']
            if isInsidePolygon(sku_coords[sku], cj_point):  # 如果sku中心点在场景中
                in_common_cj = True
                outLayerSku_cj[sku] = cj
                if cj_layer[cj]:  # 场景中有层
                    dis = {layer_info[j]['label']: pdist([layer_coords[layer_info[j]['label']], sku_coords[sku]]) for j
                           in
                           range(len(layer_info)) if
                           layer_info[j]['label'] in cj_layer[cj]}  # cj_layer[cj] = ['layer-5', 'layer-6', 'layer-9']
                    sdis = sorted(dis.items(), key=lambda kv: (kv[1], kv[0]))
                    # print(sdis)
                else:  # 场景中一个层没有  这种情况注意下
                    # if not sdis:
                    dis = {layer_info[j]['label']: pdist([layer_coords[layer_info[j]['label']], sku_coords[sku]]) for j
                           in range(len(layer_info))}
                    sdis = sorted(dis.items(), key=lambda kv: (kv[1], kv[0]))
                if not sdis:  # sdis可能为[]
                    pass
                else:
                    if sdis[0][0] in exact_sku.keys():
                        exact_sku[sdis[0][0]].append(sku)
                    else:
                        exact_sku[sdis[0][0]] = [sku]
        # if not in_common_cj and not flat_cj_layer_dlsbx:  #对在双开门lsbx中但不在层中sku分配
        #     for i in range(len(flat_cj_layer_dlsbx)):  #{'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-9'],'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-8']}
        #         dis = {}
        # layer_sku[layer] = temp  #20210818 删掉这一行
    for lay, sku in exact_sku.items():
        if lay in layer_sku.keys():
            layer_sku[lay] = layer_sku[lay] + sku  # {'layer-1': ['PPPPPPP-3', 'PPPPPPP-5', 'FFFFFFF-1']}
    ###################################################################################################################

    # 加空层待定bottlesku:  ly 2021/6/9
    add_count = 1
    # pdb.set_trace()
    for l, s in layer_sku.items():
        if not s and l in layers_add_PPP:
            daidingName = 'empty_layer_sku' + '-' + str(add_count)
            daidingLoc = gen_PPP_location(layer_coords[l], layer_ratios[l], img_w, img_h)
            layer_sku[l].append(daidingName)
            skudata['boxInfo'].append({'layer': l.split('-')[-1], 'skuName': daidingName, 'location': daidingLoc,
                                       'score': 100.0})  # layerNum、layerLine、layerOut暂时不改
            sku_coords[daidingName] = [(daidingLoc['xmin'] + daidingLoc['xmax']) * 0.5 * img_w,
                                       (daidingLoc['ymin'] + daidingLoc['ymax']) * 0.5 * img_h, ]
            add_count += 1

    # 【0】对分层框进行排序
    # pdb.set_trace()
    real_layers = {}  # 【real_layers】 real_layers={'ty_cj_lsbx-1': {'layer-15': '1', 'layer-4': '2', 'layer-2': '3', 'layer-7': '4', 'layer-8': '5'}}
    real_layers_tylsbx = {}
    real_layers_shlsbx = {}
    for ele in cjClass:
        temp_layer = {}
        temp_x = {}
        temp_y = {}
        if len(cj_layer[ele]) > 1:  # 如果场景中的层数大于1
            for layer in cj_layer[ele]:
                temp_x[layer] = layer_coords[layer][0]  # 存放层的中心点x
                temp_y[layer] = layer_coords[layer][1]  # 存放层的中心点y

            sorted_x = sorted(temp_x.items(), key=lambda x: x[1])
            sorted_y = sorted(temp_y.items(), key=lambda x: x[1])
            if ratio_count > max(1, 0.4 * len(layer_coords)):  #
                for i in range(len(cj_layer[ele])):
                    temp_layer[sorted_x[i][0]] = str(i + 1)
            else:
                for i in range(len(cj_layer[ele])):
                    temp_layer[sorted_y[i][0]] = str(i + 1)
            real_layers[ele] = temp_layer
        elif len(cj_layer[ele]) == 1:  # 如果场景中的层数为1，则所有层数为1
            temp_layer[cj_layer[ele][0]] = str(1)
            real_layers[
                ele] = temp_layer  # real_layers={'ty_cj_lsbx-1': {'layer-5': '1', 'layer-1': '2', 'layer-3': '3', 'layer-2': '4', 'layer-4': '5'}, 'ty_cj_lsbx-2': {'layer-7': '1', 'layer-6': '2'}}

    cnt_single_shlsbx = 0  # 计算单门立式冰箱数量
    cnt_single_tylsbx = 0  # 计算单门立式冰箱数量
    for cjName in cj_layer.keys():
        if cjName.split('-')[0] == 'ty_lsbx_cj':
            cnt_single_tylsbx += 1
        elif cjName.split('-')[0] == 'ty_cj_lsbx':
            cnt_single_shlsbx += 1
    # print(cnt_single_tylsbx, cnt_single_shlsbx)

    # 【1】对ty双开门冰箱门排序
    for i in range(len(cj_dlsbx_info)):
        ele = cj_dlsbx_info[i]
        if ratio_count > max(1, 0.4 * len(layer_coords)):
            ele_sort = sorted(ele, key=lambda x: compute_centroid(x['points'])[1])  # 旋转
        else:
            ele_sort = sorted(ele, key=lambda x: compute_centroid(x['points'])[0])  # 正常
        cj_dlsbx_info[i] = [k['label'] for k in ele_sort]  # cj_dlsbx_info=[['ty_lsbx_cj-1', 'ty_lsbx_cj-2']]
    cj_dlsbx_info = [i for j in cj_dlsbx_info for i in j]
    # 新增信息
    cnt = 1 + cnt_single_tylsbx  # 这里可能要从len(单门tylsbx)开始排序
    cj_layer_dlsbx_merge = []  # [{'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9', 'layer-2', 'layer-4', 'layer-6', 'layer-8']}]
    dlsbx_merge = {}  # {'ty_lsbx_cj-1': ['ty_lsbx_cj-1', 'ty_lsbx_cj-2']}
    for ele in cj_layer_dlsbx:
        ele_m = {}  # 'm' for 'merge'
        lay_m = []
        lsbx_m = []
        for cj in cj_dlsbx_info:
            if cj in ele.keys():
                # for cj,layers in ele.items():
                #     if cj.rsplit('-',1)[0] == 'ty_lsbx_cj':
                #         lay_m += layers
                lay_m += ele[cj]
                lsbx_m.append(cj)
        ele_m['ty_lsbx_cj-' + str(cnt)] = lay_m
        cj_layer_dlsbx_merge.append(ele_m)
        dlsbx_merge['ty_lsbx_cj-' + str(cnt)] = lsbx_m
        cnt += 1

    cj_mergecj = {}
    for mergecj, cjlist in dlsbx_merge.items():
        for cj in cjlist:
            cj_mergecj[cj] = mergecj

    # 对双开门ty冰箱分层框排序
    # pdb.set_trace();print('对ty双开门冰箱分层框排序')
    # for ele in cjDlsbxClass:
    # cj_layer_dlsbx -> cj_layer_dlsbx_merge ,先排序才能merge
    for ele in cj_layer_dlsbx:  # cj_layer_dlsbx = [ {'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9'], 'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-6', 'layer-8']} ]
        layer_count = 0
        for cj in cj_dlsbx_info:
            if cj in ele.keys():
                layers = ele[cj]
            # for cj,layers in ele.items():
            temp_layer = {}
            temp_x = {}
            temp_y = {}
            if len(layers) > 1:  # 如果场景中的层数大于1
                for layer in layers:
                    temp_x[layer] = layer_coords[layer][0]  # 存放层的中心点x
                    temp_y[layer] = layer_coords[layer][1]  # 存放层的中心点y

                sorted_x = sorted(temp_x.items(), key=lambda x: x[1])
                sorted_y = sorted(temp_y.items(), key=lambda x: x[1])
                if ratio_count > max(1, 0.4 * len(layer_coords)):  # 旋转
                    for i in range(len(layers)):
                        temp_layer[sorted_x[i][0]] = str(i + 1 + layer_count)
                else:
                    for i in range(len(layers)):
                        temp_layer[sorted_y[i][0]] = str(i + 1 + layer_count)
                real_layers_tylsbx[cj] = temp_layer
                layer_count = i + 1
            elif len(layers) == 1:  # 如果场景中的层数为1，则所有层数为1
                temp_layer[layers[0]] = str(1 + layer_count)
                real_layers_tylsbx[cj] = temp_layer
                layer_count = i + 1

    # 【2】对sh双开门冰箱门排序
    for i in range(len(cj_shlsbx_info)):
        ele = cj_shlsbx_info[i]
        if ratio_count > max(1, 0.4 * len(layer_coords)):
            ele_sort = sorted(ele, key=lambda x: compute_centroid(x['points'])[1])  # 旋转
        else:
            ele_sort = sorted(ele, key=lambda x: compute_centroid(x['points'])[0])  # 正常
        cj_shlsbx_info[i] = [k['label'] for k in ele_sort]  # cj_dlsbx_info=[['ty_lsbx_cj-1', 'ty_lsbx_cj-2']]
    cj_shlsbx_info = [i for j in cj_shlsbx_info for i in j]
    # pdb.set_trace()
    # 新增信息
    cnt = 1 + cnt_single_shlsbx  # 这里可能要从len(单门lsbx开始排序)
    cj_layer_shlsbx_merge = []  # [{'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9', 'layer-2', 'layer-4', 'layer-6', 'layer-8']}]
    shlsbx_merge = {}  # {'ty_lsbx_cj-1': ['ty_lsbx_cj-1', 'ty_lsbx_cj-2']}
    for ele in cj_layer_shlsbx:
        ele_m = {}  # 'm' for 'merge'
        lay_m = []
        lsbx_m = []
        for cj in cj_shlsbx_info:
            if cj in ele.keys():
                # for cj,layers in ele.items():
                #     if cj.rsplit('-',1)[0] == 'ty_lsbx_cj':
                #         lay_m += layers
                lay_m += ele[cj]
                lsbx_m.append(cj)
        ele_m['ty_cj_lsbx-' + str(cnt)] = lay_m
        cj_layer_shlsbx_merge.append(ele_m)
        shlsbx_merge['ty_cj_lsbx-' + str(cnt)] = lsbx_m  # 'sh_cj_lsbx-'
        cnt += 1

    cj_mergecj_sh = {}  # {'ty_cj_lsbx-1': 'ty_cj_lsbx-1', 'ty_cj_lsbx-2': 'ty_cj_lsbx-1'},如果cnt排序累积则{'ty_cj_lsbx-1': 'ty_cj_lsbx-2', 'ty_cj_lsbx-2': 'ty_cj_lsbx-2'}
    for mergecj, cjlist in shlsbx_merge.items():
        for cj in cjlist:
            cj_mergecj_sh[cj] = mergecj  # 【cj_mergecj_sh】每个lsbx门对应合并后的lsbx门

    # 对双开门sh冰箱分层框排序
    # pdb.set_trace();print('对双开门冰箱分层框排序')
    # for ele in cjDlsbxClass:
    # cj_layer_dlsbx -> cj_layer_dlsbx_merge ,先排序才能merge
    for ele in cj_layer_shlsbx:  # cj_layer_dlsbx = [ {'ty_lsbx_cj-1': ['layer-1', 'layer-3', 'layer-5', 'layer-7', 'layer-9'], 'ty_lsbx_cj-2': ['layer-2', 'layer-4', 'layer-6', 'layer-8']} ]
        layer_count = 0
        for cj in cj_shlsbx_info:
            if cj in ele.keys():
                layers = ele[cj]
            # for cj,layers in ele.items():
            temp_layer = {}
            temp_x = {}
            temp_y = {}
            if len(layers) > 1:  # 如果场景中的层数大于1
                for layer in layers:
                    temp_x[layer] = layer_coords[layer][0]  # 存放层的中心点x
                    temp_y[layer] = layer_coords[layer][1]  # 存放层的中心点y

                sorted_x = sorted(temp_x.items(), key=lambda x: x[1])
                sorted_y = sorted(temp_y.items(), key=lambda x: x[1])
                if ratio_count > max(1, 0.4 * len(layer_coords)):  # 旋转
                    for i in range(len(layers)):
                        temp_layer[sorted_x[i][0]] = str(i + 1 + layer_count)
                else:
                    for i in range(len(layers)):
                        temp_layer[sorted_y[i][0]] = str(i + 1 + layer_count)
                real_layers_shlsbx[cj] = temp_layer
                layer_count += len(layers)  # layer_count = i + 1
            elif len(layers) == 1:  # 如果场景中的层数为1，则所有层数为1
                temp_layer[layers[0]] = str(1 + layer_count)
                real_layers_shlsbx[cj] = temp_layer
                layer_count += len(layers)  # layer_count = i + 1

    # pdb.set_trace();print('sku_c')
    sku_c = {}  # 原先使用sku_c，后改为sku_cj
    if real_layers:  # real_layers = {'ty_lsbx_cj-1': {'layer-1': '1', 'layer-9': '2', 'layer-7': '3', 'layer-5': '4', 'layer-3': '5'}, 'ty_lsbx_cj-2': {'layer-2': '6', 'layer-4': '7', 'layer-6': '8', 'layer-8': '9'}}
        # sku对应场景
        real_layer = reduce(lambda x, y: dict(x, **y),
                            real_layers.values())  # real_layer={'layer-5': '1', 'layer-6': '2', 'layer-9': '3', 'layer-7': '1', 'layer-4': '2', 'layer-2': '3', 'layer-3': '4', 'layer-1': '5', 'layer-8': '6'
    if real_layers_tylsbx:
        real_layer_ty = reduce(lambda x, y: dict(x, **y),
                               real_layers_tylsbx.values())  # real_layer={'layer-5': '1', 'layer-6': '2', 'layer-9': '3', 'layer-7': '1', 'layer-4': '2', 'layer-2': '3', 'layer-3': '4', 'layer-1': '5', 'layer-8': '6'
    if real_layers_shlsbx:
        real_layer_sh = reduce(lambda x, y: dict(x, **y),
                               real_layers_shlsbx.values())  # real_layer={'layer-5': '1', 'layer-6': '2', 'layer-9': '3', 'layer-7': '1', 'layer-4': '2', 'layer-2': '3', 'layer-3': '4', 'layer-1': '5', 'layer-8': '6'
    real_layer_all = dict()
    try:
        real_layer_all.update(real_layer)
    except:
        pass
    try:
        real_layer_all.update(real_layer_ty)
    except:
        pass
    try:
        real_layer_all.update(real_layer_sh)
    except:
        pass
    real_layer = real_layer_all

    if real_layers:  # 原先计算sku_c的部分。
        for k, v in real_layers.items():  # {'ty_cj_hj-1': {'layer-1': '1'}}
            for layer, skus in layer_sku.items():  # {'layer-1': ['PPPPPPP-3', 'PPPPPPP-5', 'FFFFFFF-1']}
                if layer in v:
                    for sku in skus:  # lsbx2如果没有层，其中的sku会被放入lsbx1的层，进而放入lsbx1这个场景
                        # sku_c[sku] = k  #{'daiding_101-2': 'ty_cj_clj-1', 'daiding_101-1': 'ty_cj_clj-1', 'daiding_101-3': 'ty_cj_clj-1', '0003730-1': 'ty_cj_hj-1'}
                        sku_c[sku] = cj_mergecj[k] if k in cj_mergecj.keys() else k  # 统一多门冰箱
                        # cj_mergecj_sh
                        sku_c[sku] = cj_mergecj_sh[k] if k in cj_mergecj_sh.keys() else k  # sh多门冰箱 可能有问题

    # 对sku和最终分层结果合并
    # 存在一个bug 如果没有 layer，那么res_fc 就为空{} 特别是陈列挂架的情况` 或者近距离货架有cj没layer | res_fc={}会怎样? | 或许已解决
    res = {}
    res_fc = {}  # SKU对应层号：{'0000433-1': '4', '0000433-2': '4', '0000433-3': '4', '0003170-1': '3', '0003170-2': '3', '0003303-1': '3'}
    try:
        for layer in layerClass:  # layerClass:['layer-1', 'layer-2', 'layer-3', 'layer-4', 'layer-5', 'layer-6', 'layer-7', 'layer-8', 'layer-9'] #所有层
            if layer_sku[layer] is not None:
                # if layer_sku[layer].startswith('PPPPPPP_add'):
                #     continue
                for sku in layer_sku[layer]:
                    if layer in real_layer:  # real_layer:在场景中的层
                        res_fc[sku] = real_layer[
                            layer]  # {'PPPPPPP_add-1': '1', 'PPPPPPP_add-2': '6', 'PPPPPPP_add-3': '5', 'PPPPPPP_add-4': '7', '0000433-1': '4', '0000433-2': '4', '0000433-3': '4', 'PPPPPPP_add-5': '8', '0003170-1': '3', '0003170-2': '3', '0003303-1': '3', 'PPPPPPP_add-6': '9', 'PPPPPPP_add-7': '2'}
                    else:
                        res_fc[sku] = 0
    except:
        pass

    # 对每一个SKU遍历，那么就可以进行操作了
    # pdb.set_trace()
    for i in range(len(sku_info)):
        try:
            res_fc[sku_info[i]['skuName']]  # 即判断每个sku: if '0003170-1' in res_fc.keys()
        except:
            # 用于计算没有在层内的sku的相关信息
            # pdb.set_trace();print('-----------------')
            # print('*outside layer*', sku_info[i]['skuName'])
            res_fc[sku_info[i]['skuName']] = 0  # 初始化为没有在层内的所有的SKU层数都为0
            sku_c[sku_info[i]['skuName']] = 'ty_cj_others'  # 没有在层内的SKU所属场景置为others  ????
            '''
            # 用于计算陈列挂架的相关信息
            if 'cj_clgj' in scene_info['sceneSet']:
                for k in range(len(cj_info)):
                    cj = cj_info[k]['label'].split('-')[0]
                    if cj == 'cj_clgj': 
                       cj_point = cj_info[k]['points']
                       isInClgj = isInsidePolygon(sku_coords[sku_info[i]['skuName']], cj_point)
                       skuName = sku_info[i]['skuName'].split('-')[0]
                       
                       if isInClgj and skuName in clgjSkuList:
                          res_fc[sku_info[i]['skuName']] = 1
            '''
            if scene_info['sceneSetNum'] == 1:  # and 'cj_hj' in scene_info['sceneSet']:
                for k in range(len(cj_info)):
                    cj = cj_info[k]['label']  # 'ty_lsbx_cj-1'
                    cj_point = cj_info[k]['points'];
                    dist_temp = {}
                    if len(cj_layer[cj]) == 0:
                        if isInsidePolygon(sku_coords[sku_info[i]['skuName']], cj_point):  # 存在bug
                            res_fc[sku_info[i]['skuName']] = 1  # 重新把0层置为1层？
                    else:
                        if isInsidePolygon(sku_coords[sku_info[i]['skuName']], cj_point):
                            for layer in cj_layer[cj]:
                                if ratio_count > max(1, 0.4 * len(layer_coords)):  # 如果图片旋转
                                    dist_temp[layer] = np.sqrt((sku_coords[sku_info[i]['skuName']][0] -
                                                                layer_coords[layer][0]) ** 2)  # 计算离sku x坐标最近的层
                                else:
                                    dist_temp[layer] = np.sqrt(
                                        (sku_coords[sku_info[i]['skuName']][1] - layer_coords[layer][1]) ** 2)
                            sorted_dist = sorted(dist_temp.items(), key=lambda x: x[1])
                            res_fc[sku_info[i]['skuName']] = real_layer[sorted_dist[0][0]]

            else:  # 有个问题，当场景数多于1，那不在场景中的sku就不管了；当场景数等于1，不在场景中的sku也分配了？
                for m in range(len(cj_info)):  # 遍历场景
                    cj = cj_info[m]['label']  # 'ty_lsbx_cj-1'
                    # cj_point = point_trans(cj_info[m], img_w, img_h)
                    cj_point = cj_info[m]['points']  # [[726, 491], [694, 1372], [1014, 1372], [1079, 491]]
                    dist_temp = {}
                    if len(cj_layer[cj]) == 0:
                        if isInsidePolygon(sku_coords[sku_info[i]['skuName']], cj_point):
                            res_fc[sku_info[i]['skuName']] = 1  # 这个场景没有层，则在里面的sku都设为1层
                    elif isInsidePolygon(sku_coords[sku_info[i]['skuName']], cj_point):  # 看sku是否在场景中
                        for layer in cj_layer[cj]:
                            if ratio_count > max(1, 0.4 * len(layer_coords)):
                                dist_temp[layer] = np.sqrt(
                                    (sku_coords[sku_info[i]['skuName']][0] - layer_coords[layer][0]) ** 2)
                            else:
                                dist_temp[layer] = np.sqrt(
                                    (sku_coords[sku_info[i]['skuName']][1] - layer_coords[layer][1]) ** 2)
                        sorted_dist = sorted(dist_temp.items(), key=lambda x: x[1])
                        res_fc[sku_info[i]['skuName']] = real_layer[sorted_dist[0][0]]

    # pdb.set_trace()

    ############
    # sku_cj has something wrong
    sku_other = [k for k, v in res_fc.items() if v == 0]  # []
    sku_c.update({sku: 'ty_cj_others' for sku in
                  sku_other})  # {'PPPPPPP-3': 'ty_cj_hj-1', 'PPPPPPP-5': 'ty_cj_hj-1', 'FFFFFFF-1': 'ty_cj_hj-1', 'PPPPPPP-1': 'ty_cj_others', 'PPPPPPP-2': 'ty_cj_others', 'PPPPPPP-4': 'ty_cj_others', 'ty_xz_daiding-1': 'ty_cj_others', '0001787-1': 'ty_cj_others'}
    sku_layer = {}  # 【SKU-层】
    sku_cj = {}  # 【SKU-场景】  #{'0003170-1': 'ty_lsbx_cj-2', '0003170-2': 'ty_lsbx_cj-2'}
    # {'PPPPPPP-3': 'ty_cj_hj-1', 'PPPPPPP-5': 'ty_cj_hj-1', 'FFFFFFF-1': 'ty_cj_hj-1', 'PPPPPPP-1': 'ty_cj_hj-1', 'PPPPPPP-2': 'ty_cj_hj-1', 'PPPPPPP-4': 'ty_cj_hj-1', 'ty_xz_daiding-1': 'ty_cj_hj-1', '0001787-1': 'ty_cj_hj-1'}
    cj_layer_dlsbx_flat = reduce(lambda x, y: dict(x, **y), cj_layer_dlsbx_merge) if cj_layer_dlsbx_merge else []
    cj_layer_shlsbx_flat = reduce(lambda x, y: dict(x, **y), cj_layer_shlsbx_merge) if cj_layer_shlsbx_merge else []
    # pdb.set_trace();print('real_layer')
    layer_points_map = {}
    for layer in layer_info:
        layer_points_map[layer['label']] = layer['points']

    if 'real_layer' in locals().keys():
        for sku_ in res_fc:  # 【遍历sku】 # {'0000446-1': '2', '0000446-2': '2'}
            for layer_ in real_layer:  # 【遍历layer】 # {'layer-5': '1', 'layer-1': '2', 'layer-3': '3', 'layer-2': '4', 'layer-4': '5', 'layer-7': '1', 'layer-6': '2'
                if res_fc[sku_] == real_layer[layer_] and \
                        isInsidePolygon(sku_coords[sku_],
                                        layer_points_map[layer_]):  # 这里，可能layer-5和layer-7是两个场景中的第’1‘层，导致sku错乱
                    # # pdb.set_trace()
                    sku_layer[sku_] = layer_
                    for cj in cj_layer:  # 【遍历cj】 # {'ty_cj_lsbx-1': ['layer-1', 'layer-2', 'layer-3', 'layer-4', 'layer-5'], 'ty_cj_lsbx-2': ['layer-6', 'layer-7']}
                        if sku_layer[sku_] in cj_layer[cj]:  # 如果sku所在的层在场景包含的层中，就把sku和这个场景对应
                            sku_cj[sku_] = cj  # 没有包含陈列挂架场景
                        # else: #加上这个
                        #     sku_cj[sku_] = 'ty_cj_others'
                    # 加上double_lsbx
                    for cj in cj_layer_dlsbx_flat:
                        if sku_layer[sku_] in cj_layer_dlsbx_flat[cj]:
                            sku_cj[sku_] = cj
                        # else:  # 加上这个
                        #     sku_cj[sku_] = 'ty_cj_others'
                    # 加上sh_lsbx
                    for cj in cj_layer_shlsbx_flat:  # 相同场景名的单、多门冰箱会重叠，因此修改名
                        if sku_layer[sku_] in cj_layer_shlsbx_flat[cj]:
                            sku_cj[sku_] = cj
                        # else:  # 加上这个
                        #     sku_cj[sku_] = 'ty_cj_others'
        for sku_ in res_fc:
            if not sku_ in sku_cj:
                if sku_ in outLayerSku_cj:
                    sku_cj[sku_] = outLayerSku_cj[sku_]
                else:
                    sku_cj[sku_] = 'ty_cj_others'
    else:  # 只有场景没有layer的情况
        for sku, cj in sku_c.items():
            if sku in outLayerSku_cj.keys() and cj == 'ty_cj_others':
                sku_cj[sku] = outLayerSku_cj[sku]

    layer_count = {ele: len(cj_layer[ele]) for ele in cj_layer}
    if cj_layer_dlsbx_flat:
        layer_count_lsbx = {ele: len(cj_layer_dlsbx_flat[ele]) for ele in cj_layer_dlsbx_flat}
        layer_count.update(layer_count_lsbx)  # 两结果合并
        # layer_count[ele] = len(cj_layer[ele])
    ## pdb.set_trace()
    if cj_layer_shlsbx_flat:
        layer_count_shlsbx = {ele: len(cj_layer_shlsbx_flat[ele]) for ele in cj_layer_shlsbx_flat}
        layer_count.update(layer_count_shlsbx)  # 两结果合并
        # layer_count[ele] = len(cj_layer[ele])
    layer_count['ty_cj_others'] = 0  # {'ty_lsbx_cj-1': 5, 'ty_lsbx_cj-2': 4, 'ty_cj_others': 0}

    if_rot = True if ratio_count > max(1, 0.4 * len(layer_coords)) else False

    res['layer'] = res_fc  # sku：层号
    res['skuCj'] = sku_cj  # sku_c       #sku：场景  sku_cj曾有些问题，lmx使用sku_c返回； sku对应的'ty_cj_lsbx'是合并后的立式冰箱名
    res['layerCount'] = layer_count  # 场景：层数
    res['layerOut'] = []
    res['dlsbxMerge'] = dlsbx_merge
    res['shlsbxMerge'] = shlsbx_merge  # {'ty_cj_lsbx-2': ['ty_cj_lsbx-1', 'ty_cj_lsbx-2']}
    res['ifRot'] = if_rot
    # pdb.set_trace()
    return res


def mask_compute_all_centroid(all_info, img_h, img_w):
    ''' 计算所有多边形的中心点(这里特别指层信息)
    Args:
       all_info: 字典,所有多边形信息
       img_h, img_w: 图像的高度和宽度

    Returns:
       centroid_dict: 字典, {'layer-1':[x1, y1], 'layer-2':[x1, y1]}
    '''
    centroid_dict = {}
    for ele in all_info:
        point = ele['points']
        centroid = compute_centroid(point)
        centroid_dict[ele['label']] = centroid
    return centroid_dict


def mask_compute_ratio(all_info, img_h, img_w):
    ''' 计算mask的外接轮廓矩形的高和宽[maxy- miny, maxx-minx]

    Args:
        all_info: labelme标注的其中一个mask信息
        img_h: 图像的高度
        img_w: 图像的宽度

    Returns:
        ratio_dict: 字典：{label:[高度，宽度]}
    '''
    ratio_dict = {}
    for ele in all_info:
        point = ele['points']
        point_x = [x[0] for x in point]
        point_y = [y[1] for y in point]
        min_x, max_x, min_y, max_y = min(point_x), max(point_x), min(point_y), max(point_y)
        ratio_dict[ele['label']] = [max_y - min_y, max_x - min_x]
    return ratio_dict


def gen_PPP_location(layer_coords, layer_ratios, img_w, img_h):
    '''ly 根据层的位置增加PPPPPPP_add在原图上的位置'''
    xmin = (layer_coords[0] - 0.08 * layer_ratios[0]) / img_w
    xmax = (layer_coords[0] + 0.08 * layer_ratios[0]) / img_w
    ymin = (layer_coords[1] - 0.08 * layer_ratios[1]) / img_h
    ymax = (layer_coords[1] + 0.08 * layer_ratios[1]) / img_h
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}


def rmZeroLayer(res, main_scene):
    """ 基于冰柜的场景删掉layer==0的SKU """
    """ 会把立式冰柜之外的本品SKU信息也删掉"""

    if (len(main_scene) == 1 and "lsbx" in main_scene[0]):
        result = {}
        if int(res['layerNum']) > 0:
            boxInfo = []
            for i in range(len(res['boxInfo'])):
                if int(res['boxInfo'][i]['layer']) > 0:
                    boxInfo.append(res['boxInfo'][i])
            res['boxInfo'] = boxInfo
            result = res
        else:
            result = res
    else:
        result = res

    return result


def closePicture(results):
    # pdb.set_trace()
    '''大头贴处理，针对冰柜、热饮机场景， 结合距离、层数、sku尺寸'''
    '''近距离hj容易识别成lsbx'''
    if len(results['dataInfo']) > 2:  # 场景数
        return results
    elif len(results['dataInfo']) == 2:  # 一种特殊情况 统一企业logo检成tyryj
        if {results['dataInfo'][0]['placeSegname'], results['dataInfo'][1]['placeSegname']} == {'ty_ryj_cj',
                                                                                                'ty_lsbx_cj'}:
            datainfo = sorted(results['dataInfo'], key=lambda k: k['placeType'])  # [lsbx,ryj]
            if len(datainfo[0]['boxInfo']) > 0 and len(datainfo[1]['boxInfo']) == 0:
                if int(datainfo[0]['layerNum']) <= 3:
                    cjW, cjH = cj_mask_ratio(datainfo[0]['placeLocation'])
                    skuW, skuH = sku_max_ratio(datainfo[0]['boxInfo'])
                    if cjH / skuH / results['imageInfo']['height'] < 2.8:
                        # print('近距离lsbx')
                        datainfo[0]['placeSegname'] = 'ty_cj_others'
                        datainfo[0]['placeType'] = 0
                        results['dataInfo'] = [datainfo[0]]
        return results
    # 只识别到一种场景：
    if results['dataInfo'][0]['placeSegname'] in ['ty_cj_others']:
        if len(results['dataInfo'][0]['boxInfo']):
            skuW, skuH = sku_max_ratio(results['dataInfo'][0]['boxInfo'])
            if 1 / skuH < 1.5 or 1 / skuW < 3:
                results['dataInfo'][0]['placeSegname'] = 'ty_cj_others'
                results['dataInfo'][0]['placeType'] = 0
                # print('近距离lsbx')
    elif results['dataInfo'][0]['placeSegname'] in ['ty_cj_wsbg']:
        if len(results['dataInfo'][0]['boxInfo']):
            skuW, skuH = sku_max_ratio(results['dataInfo'][0]['boxInfo'])
            if 1 / skuH < 1.2:
                results['dataInfo'][0]['placeSegname'] = 'ty_cj_others'
                results['dataInfo'][0]['placeType'] = 0
                # print('近距离lsbx')
    elif results['dataInfo'][0]['placeSegname'] in ['ty_cj_lsbx', 'ty_cj_ryj', 'ty_lsbx_cj', 'ty_ryj_cj']:
        if results['imageInfo']['distance'] <= 0.87 and results['imageInfo']['distance'] != 0. and len(
                results['dataInfo'][0]['boxInfo']):
            results['dataInfo'][0]['placeSegname'] = 'ty_cj_others'  # 改变场景名
            results['dataInfo'][0]['placeType'] = 0
            # print('近距离lsbx')
        elif results['imageInfo']['distance'] < 3 or results['imageInfo']['distance'] == 0.:  # 3 包括了绝大多数图片
            # if results['dataInfo'][0]['placeSegname'] == 'ty_cj_wsbg' or int(results['dataInfo'][0]['layerNum']) >=3:
            if int(results['dataInfo'][0]['layerNum']) == 1:  # 如果一层，则空场景
                # threH,threW = 3.6, 3.6
                results['dataInfo'][0]['placeSegname'] = 'ty_cj_others'
                results['dataInfo'][0]['placeType'] = 0
                # print('近距离lsbx')
                return results
            elif int(results['dataInfo'][0]['layerNum']) > 3:  # 层数超过3层，则不是空场景
                return results
            else:  # 层数为2/3
                threH, threW = 3.15, 4
                if len(results['dataInfo'][0]['boxInfo']) != 0:
                    # skuWList, skuHList = [],[]
                    # # imgW, imgH = results['imageInfo']['width'], results['imageInfo']['height']
                    # for sku in results['dataInfo'][0]['boxInfo']:
                    #     skuWList.append(sku['location']['xmax']-sku['location']['xmin'])
                    #     skuHList.append(sku['location']['ymax']-sku['location']['ymin'])
                    skuW, skuH = sku_max_ratio(results['dataInfo'][0]['boxInfo'])
                    if 1 / skuH < threH or 1 / skuW < threW:
                        results['dataInfo'][0]['placeSegname'] = 'ty_cj_others'
                        results['dataInfo'][0]['placeType'] = 0
                        # print('近距离lsbx')
    return results


def sku_max_ratio(skulist):
    skuWList, skuHList = [], []
    for sku in skulist:
        skuWList.append(sku['location']['xmax'] - sku['location']['xmin'])
        skuHList.append(sku['location']['ymax'] - sku['location']['ymin'])
    skuW, skuH = max(skuWList), max(skuHList)
    return skuW, skuH


def cj_mask_ratio(location):
    x, y = [coor[0] for coor in location], [coor[1] for coor in location]
    w = max(x) - min(x)
    h = max(y) - min(y)
    return w, h
