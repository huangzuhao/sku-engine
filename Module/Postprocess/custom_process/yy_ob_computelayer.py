from Module.builder import POSTPROCESSES
from .general_ob_computelayer import General_ob_computelayer
from .general_ob_sku import General_ob_sku
from Module.Postprocess.common.common_method import between_class_nms_by_score_union, between_class_nms_by_MinArea
import copy
from Module.Postprocess.common.compute_layer import stackSKUProcRaw, stackSKUProcSpecial, filterSKUByLayer, \
    filterSKUByLayerFine, ConvertLayers, skuStatus, SKU, getskuList
import numpy as np


@POSTPROCESSES.register_module()
class Yy_ob_computelayer(General_ob_computelayer):
    def between_class_nms(self, skulist):
        length = len(skulist)
        if length > 0:
            return yy_repocess(skulist)
        else:
            return []

    def merge_sku_cjfc(self, result_processed, width, height):
        # skudata = self.format_bbox_result(skulist, width, height)
        imageinfo = {"width": width, "height": height, "distance": 1.0, "isVision": 0}
        skulist = result_processed["sku"]
        boxeslist = result_processed["boxes"]

        skukit = SKU(skulist, imageinfo)
        boxkit = SKU(boxeslist, imageinfo)

        skudata = skukit.getskudict()
        boxdata = boxkit.getskudict()

        if self.stack_config["Enable"]:
            if self.stack_config["Func"] == "Raw":
                skudata = stackSKUProcRaw(skudata)
            elif self.stack_config["Func"] == "Special":
                skudata = stackSKUProcSpecial(skudata, self.stack_config)

        skudata = yyBoxesMerge(skudata, boxdata)
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


@POSTPROCESSES.register_module()
class Yy_ob_sku(General_ob_sku):
    def between_class_nms(self, skulist):
        length = len(skulist)
        if length > 0:
            return yy_repocess(skulist)
        else:
            return []


def yy_repocess(tempList1):
    # 2021.8
    tempList = copy.deepcopy(tempList1)
    clmList, skuList = [], []
    for item in tempList:
        if item[-1] == 'yl_yy_clm':
            clmList.append(list(item))
        else:
            skuList.append(list(item))
    if len(clmList) > 1:
        clmList = between_class_nms_by_MinArea(clmList, threshold=0.8)

    if len(skuList) > 1:
        skuList = between_class_nms_by_score_union(skuList, threshold=0.93)
    tempList = skuList + clmList
    # tempList = [list(c) for c in tempList]
    skuNameList = [i[5] for i in tempList]
    # print('**',skuNameList)
    if 'yl_yy_lght_stdK' in skuNameList:  # stdK 已优化
        if 'yl_yy_lght_dtjpx240X12' in skuNameList:  # 优先240
            name_stdK = 'yl_yy_lght_dtjpx240X12_stdK'
        elif 'yl_yy_lght_tjlz270X12' in skuNameList:
            name_stdK = 'yl_yy_lght_tjlz270X12_stdK'
        else:
            name_stdK = 'yl_yy_lght_jpx270X12_stdK'
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_stdK':
                tempList[i][5] = name_stdK

    # 合并jyx240,jxx240为jpx240
    if 'yl_yy_lght_jyx240X20' in skuNameList or 'yl_yy_lght_jxx240X20' in skuNameList:
        for i in range(len(tempList)):
            if tempList[i][5] in ['yl_yy_lght_jyx240X20', 'yl_yy_lght_jxx240X20']:
                tempList[i][5] = 'yl_yy_lght_dtjpx240X20'
            if skuNameList[i] in ['yl_yy_lght_jyx240X20', 'yl_yy_lght_jxx240X20']:
                skuNameList[i] = 'yl_yy_lght_dtjpx240X20'

    # 合并大小养生为数量多的，一样时为大养生
    if 'yl_yy_lght_dtzhys240X15' in skuNameList and 'yl_yy_lght_zhys180X20' in skuNameList:
        numList = [skuNameList.count('yl_yy_lght_dtzhys240X15'), skuNameList.count('yl_yy_lght_zhys180X20')]
        reproYS = ['yl_yy_lght_dtzhys240X15', 'yl_yy_lght_zhys180X20'][numList.index(max(numList))]
        for i in range(len(tempList)):
            if tempList[i][5] in ['yl_yy_lght_dtzhys240X15', 'yl_yy_lght_zhys180X20']:
                tempList[i][5] = reproYS
            if skuNameList[i] in ['yl_yy_lght_dtzhys240X15', 'yl_yy_lght_zhys180X20']:
                skuNameList[i] = reproYS

    # stdC
    if 'yl_yy_lght_stdC' in skuNameList:
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_stdC':
                tempList[i][5] = 'yl_yy_lght_dtjpx240X20_stdC'

    # stdGK
    if 'yl_yy_lght_stdGK' in skuNameList:
        numList = [skuNameList.count('yl_yy_lght_dtjpx240X20'), skuNameList.count('yl_yy_lght_dtzhys240X15')]
        reproSku = ['yl_yy_lght_dtjpx240X20_stdGK', 'yl_yy_lght_dtzhys240X15_stdGK'][numList.index(max(numList))]
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_stdGK':
                tempList[i][5] = reproSku

    # 养生手提袋
    if 'yl_yy_lght_stdYS' in skuNameList:  # stdYS  这个客户希望能直接识别规格，暂用旧逻辑
        # pdb.set_trace()
        num_zhys = skuNameList.count('yl_yy_lght_zhys180X20')
        num_dtzhys = skuNameList.count('yl_yy_lght_dtzhys240X15')
        # # print('numListYS:',[num_zhys,num_dtzhys])
        if num_dtzhys >= num_zhys:
            reproSku = 'yl_yy_lght_stdYS240X15'
        else:
            reproSku = 'yl_yy_lght_stdYS180X20'
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_stdYS':
                tempList[i][5] = reproSku

    # 牛运罐六个核桃箱
    if 'yl_yy_lght_nyg_240X20' in skuNameList:
        # nygmax = [skuNameList.count('yl_yy_lght_dtjpx240X20'), skuNameList.count('yl_yy_lght_jyx240X20'),
        #            skuNameList.count('yl_yy_lght_jxx240X20')]
        # suffix = ['_jpx', '_jyx', '_jxx']
        # reproSku = 'yl_yy_lght_nyg_240X20' + suffix[nygmax.index(max(nygmax))]
        # reproSku = ['yl_yy_lght_dtjpx240X20','yl_yy_lght_jyx240X20','yl_yy_lght_jxx240X20'][nygmax.index(max(nygmax))]
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_nyg_240X20':
                tempList[i][5] = 'yl_yy_lght_dtjpx240X20'

    if 'yl_yy_lght_sidejpjyjx240X20' in skuNameList:  # jpjyjx箱侧
        # sidemax = [skuNameList.count('yl_yy_lght_dtjpx240X20'), skuNameList.count('yl_yy_lght_jyx240X20'),
        #            skuNameList.count('yl_yy_lght_jxx240X20')]
        # reproSku = ['yl_yy_lght_dtjpx240X20','yl_yy_lght_jyx240X20','yl_yy_lght_jxx240X20'][sidemax.index(max(sidemax))]
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_sidejpjyjx240X20':
                tempList[i][5] = 'yl_yy_lght_dtjpx240X20'

    if 'yl_yy_lght_mc+250' in skuNameList:  # 抹茶
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_yy_lght_mc+250':
                tempList[i][5] = 'yl_yy_lght_mc+250X6X4'

    if 'yl_yy_lght_tjlzxrl240D16' in skuNameList:  # 杏仁露 已优化
        num_xrlx16 = skuNameList.count('yl_yy_lght_tjlzxrl240X16')
        num_xrlx20 = skuNameList.count('yl_yy_lght_tjlzxrl240X20')
        # print('num tjlzxlr:',[num_xrlx16,num_xrlx20])
        if num_xrlx16 == 0 or (num_xrlx16 != 0 and num_xrlx20 != 0):
            for i in range(len(tempList)):
                if tempList[i][5] == 'yl_yy_lght_tjlzxrl240D16':
                    tempList[i][5] = 'yl_yy_lght_tjlzxrl240D20'  # 优先240*20
        else:  # 即 num_xrlx20 == 0:
            for i in range(len(tempList)):
                if tempList[i][5] == 'yl_yy_lght_tjlzxrl240D20':
                    tempList[i][5] = 'yl_yy_lght_tjlzxrl240D16'

    # hongniu
    if 'yl_hn_250P' in skuNameList:
        numHnList = [skuNameList.count('yl_hn_wssgn250P'), skuNameList.count('yl_hn_wssfw250P'),
                     skuNameList.count('yl_hn_anj250P')]
        reproSku = ['yl_hn_wssgn250P', 'yl_hn_wssfw250P', 'yl_hn_anj250P'][numHnList.index(max(numHnList))]
        for i in range(len(tempList)):
            if tempList[i][5] == 'yl_hn_250P':
                tempList[i][5] = reproSku

    # 罐装jy jx -> jp
    if 'yl_yy_lght_jyx240P' in skuNameList or 'yl_yy_lght_jxx240P' in skuNameList:
        for i in range(len(tempList)):
            if tempList[i][5] in ['yl_yy_lght_jyx240P', 'yl_yy_lght_jxx240P']:
                tempList[i][5] = 'yl_yy_lght_dtjpx240P'

    return [list(l) for l in tempList]


def yyBoxesMerge(skuDet_skus, boxDet_boxes):  # 2021/12/14
    # skuDet_skus: sku、clm、bottle
    # boxDet_boxes: box、logo
    if not boxDet_boxes:
        return skuDet_skus

    skuList, bottleList, boxList, logoList = [], [], [], []
    for key,sku in skuDet_skus.items():
        if sku['skuName'] == 'daiding_101':
            bottleList.append([sku['skuMinx'],sku['skuMiny'],sku['skuMaxx'],sku['skuMaxy'],sku['id']])
        elif sku['skuName'] != 'yl_yy_clm':
            skuList.append([sku['skuMinx'],sku['skuMiny'],sku['skuMaxx'],sku['skuMaxy'],sku['id']])
    for key,box in boxDet_boxes.items():
        if box['skuName'] == 'daidingXz':
            boxList.append([box['skuMinx'],box['skuMiny'],box['skuMaxx'],box['skuMaxy'],box['id']])
        elif box['skuName'] == 'yyLogo':
            logoList.append([box['skuMinx'],box['skuMiny'],box['skuMaxx'],box['skuMaxy'],box['id']])
    # print('****')
    # if not skuList or not bottleList or not boxList or not logoList:
    #     return skuDet_skus
    if not logoList:
        return skuDet_skus
    elif not bottleList and not boxList:
        return skuDet_skus
    # print(skuList, bottleList, boxList, logoList)
    logoArray = np.array(logoList)   #所有养元Logo
    boxesArray = np.array(boxList) if boxList else np.empty((0,5))        #箱子待定模型
    skuArray = np.array(skuList) if skuList else np.empty((0,5))          #识别到sku的坐标
    bottleArray = np.array(bottleList) if bottleList else np.empty((0,5))    #识别到sku的坐标

    # pdb.set_trace()
    # print('logoArray %s, boxesArray %s, skuArray %s, bottleArray %s'%
    #       (logoArray.shape,boxesArray.shape,skuArray.shape,bottleArray.shape))

    # bottle、box，去除与box重叠的bottle
    iou_BotBox = bbox_iou_matrix(boxesArray,bottleArray)
    iou_BotBox[iou_BotBox < 0.6] = 0
    bottleInd = np.where(~iou_BotBox.any(axis=1))[0]
    bottleIndDel = np.where(iou_BotBox.any(axis=1))[0]
    bottleIndDelID = []
    for i in bottleIndDel:
        bottleIndDelID.append(bottleArray[i][-1])
    bottleArray = bottleArray[bottleInd]  #这一步要在bottleIndDelID之后

    # bottle、logo，保留与logo重叠的bottle, 删除与bottle重叠的logo
    iou_BotLogo = bbox_iou_matrix(logoArray,bottleArray)
    iou_BotLogo[iou_BotLogo < 0.8] = 0
    yyBottleInd = np.where(iou_BotLogo.any(axis=1))[0]
    # yyBottleArray = bottleArray[yyBottleInd]
    for i in yyBottleInd:
        skuDet_skus[bottleArray[i][-1]]['skuName'] = 'yyBottle'
    logoArrayInd = np.where(~iou_BotLogo.T.any(axis=1))[0]
    logoArray = logoArray[logoArrayInd]

    # sku、box，去除与sku重叠的box
    iou_M = bbox_iou_matrix(skuArray,boxesArray)
    iou_M[iou_M < 0.4] = 0
    othersBoxInd = np.where(~iou_M.any(axis=1))[0]
    othersBox = boxesArray[othersBoxInd]  #非养元box

    # box、logo，保留与logo重叠的box
    iou_N = bbox_iou_matrix(logoArray,othersBox)
    iou_N[iou_N < 0.8] = 0
    yyBoxInd = np.where(iou_N.any(axis=1))[0]  #养元box的index

    if skuDet_skus:
        maxSkusInd = max(skuDet_skus.keys()) + 1
    else:
        maxSkusInd = 0

    yyBoxList = []
    for i in yyBoxInd:
        yyBox = {maxSkusInd : boxDet_boxes[boxList[othersBoxInd[i]][-1]]}  #待定箱子 'daidingXz' -> 'yyBox'
        yyBox[maxSkusInd]['skuName'] = 'yyBox'
        skuDet_skus.update(yyBox)
        maxSkusInd += 1

    #删除与box重叠的bottle
    for i in bottleIndDelID:
        if skuDet_skus[i]['skuName'] == 'daiding_101': #这句判断应该是多余的
            del skuDet_skus[i]

    return skuDet_skus



def bbox_iou_matrix(bboxes1, bboxes2):  #bboxes1箱子，bboxes2是sku。 2021/12/14
        iou_matrix = np.zeros((len(bboxes2), len(bboxes1)))
        for i in range(len(bboxes2)):
            bboxes = bboxes2[i][None, :]
            bboxes = np.repeat(bboxes, len(bboxes1), axis=0)
            x1 = np.concatenate((bboxes[:, 0:1], bboxes1[:, 0:1]), axis=1).max(1)
            y1 = np.concatenate((bboxes[:, 1:2], bboxes1[:, 1:2]), axis=1).max(1)
            x2 = np.concatenate((bboxes[:, 2:3], bboxes1[:, 2:3]), axis=1).min(1)
            y2 = np.concatenate((bboxes[:, 3:4], bboxes1[:, 3:4]), axis=1).min(1)
            dw = x2 - x1
            dw[dw < 0] = 0
            dh = y2 - y1
            dh[dh < 0] = 0
            inter = dw * dh
            areas1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            areas2 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
            # union = areas1 + areas2 - inter
            union = np.concatenate((areas1[:, None], areas2[:, None]), 1).min(1)
            iou = inter / union
            iou_matrix[i] = iou
        return iou_matrix


