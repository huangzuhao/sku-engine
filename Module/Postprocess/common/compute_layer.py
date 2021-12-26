import operator


def stackSKUProcRaw(skus):
    for k, sku in skus.items():
        skus[k]['id'] = k
    tempDict = skus.copy()
    for k1, v1 in tempDict.items():

        centerX = v1['centerX']
        centerY = v1['centerY']
        skuW = v1['skuW']
        skuH = v1['skuH']
        tempDict[k1]['layerOut'].append(k1)
        for k2, v2 in tempDict.items():

            if abs(centerX - v2['centerX']) < skuW * 0.75 \
                    and abs(centerY - v2['centerY']) < skuH * 0.51 + v2['skuH'] * 0.51 \
                    and k1 != k2 \
                    and k2 not in v1['layerOut']:
                ymin = min(tempDict[k1]['skuMiny'], tempDict[k2]['skuMiny'])
                ymax = max(tempDict[k1]['skuMaxy'], tempDict[k2]['skuMaxy'])
                tempDict[k1]['skuMiny'] = ymin
                tempDict[k1]['skuMaxy'] = ymax
                tempDict[k1]['centerY'] = (ymin + ymax) / 2
                tempDict[k1]['skuH'] = ymax - ymin
                tempDict[k1]['layerOut'].append(k2)
                tempDict[k2]['skuMiny'] = ymin
                tempDict[k2]['skuMaxy'] = ymax
                tempDict[k2]['centerY'] = (ymin + ymax) / 2
                tempDict[k2]['skuH'] = ymax - ymin
                tempDict[k2]['layerOut'].append(k1)
                centerY = tempDict[k1]['centerY']
                skuH = ymax - ymin
    return tempDict


def stackSKUProcSpecial(skus, stackThr):
    """Calculate the skus should be stacked."""
    tempDict = skus.copy()
    for k1, v1 in tempDict.items():
        centerX = v1['centerX']
        centerY = v1['centerY']
        skuW = v1['skuW']
        skuH = v1['skuH']
        tempDict[k1]['layerOut'].append(k1)
        scaleW = 0.75
        scaleH = 0.59
        scaleHsku = 0.52
        # scaleHdiff = 0.08
        for k2, v2 in tempDict.items():

            centerXdiff = abs(centerX - v2['centerX'])
            centerYdiff = abs(centerY - v2['centerY'])

            # If the first sku and the second sku are most likely stacked.
            if k1 != k2 and centerXdiff < skuW * scaleW and centerYdiff < skuH * scaleH + v2['skuH'] * scaleH:
                # If the first sku and the second sku with same name.
                if v1['skuName'] == v2['skuName']:  # a_hn_201 &
                    scaleHsku = 0.58
                    # scaleHdiff = 0.10
                skuNameInConfig = False

                # if v1['skuName'] in stackThr['Specify'] and v2['skuName'] in stackThr['Specify'] and float(
                #         stackThr[v1['skuName']]) == float(stackThr[v2['skuName']]):
                #     scaleHsku = float(stackThr[v1['skuName']])
                #     # scaleHdiff = 0.20
                #     skuNameInConfig = True

                if "Specify" in stackThr and v1['skuName'] in stackThr['Specify'] and v2['skuName'] in stackThr[
                    'Specify'] and float(
                        stackThr['Specify'][v1['skuName']]) == float(stackThr['Specify'][v2['skuName']]):
                    scaleHsku = float(stackThr['Specify'][v1['skuName']])
                    # scaleHdiff = 0.20
                    skuNameInConfig = True

                if not skuNameInConfig:
                    if "Default" in stackThr and stackThr['Default'] > 0:
                        scaleHsku = float(stackThr['Default'])
                    else:
                        scaleHsku = 0.518
                    # scaleHdiff = 0.06

            if centerXdiff < skuW * scaleW \
                    and centerYdiff < skuH * scaleHsku + v2['skuH'] * scaleHsku \
                    and k1 != k2 \
                    and k2 not in v1['layerOut'] \
                    and tempDict[k1]['skuName'] == tempDict[k2]['skuName']:
                ymin = min(tempDict[k1]['skuMiny'], tempDict[k2]['skuMiny'])
                ymax = max(tempDict[k1]['skuMaxy'], tempDict[k2]['skuMaxy'])
                tempDict[k1]['skuMiny'] = ymin
                tempDict[k1]['skuMaxy'] = ymax
                tempDict[k1]['centerY'] = (ymin + ymax) / 2
                tempDict[k1]['skuH'] = ymax - ymin
                tempDict[k1]['layerOut'].append(k2)
                tempDict[k2]['skuMiny'] = ymin
                tempDict[k2]['skuMaxy'] = ymax
                tempDict[k2]['centerY'] = (ymin + ymax) / 2
                tempDict[k2]['skuH'] = ymax - ymin
                tempDict[k2]['layerOut'].append(k1)
                centerY = tempDict[k1]['centerY']
                skuH = ymax - ymin

    return tempDict


def filterSKUByLayer(skulist, thres, imgInfo):
    """The raw layer function."""
    leftIDs = getIDs(skulist)
    layerParser = layerSKU(skulist, thres)
    layerParser.getSkuAvgWH()
    layersDict = {}
    layers = []
    idx = 1
    while len(leftIDs) > 0:
        layerParser.getLayerMaxAndMinYBox()
        top_skulist, left_skulist, leftIDs, topIDs = layerParser.skufilter()
        layer = Layer(imgInfo)
        layer.getBottomLine(top_skulist)
        layersDict[idx] = layer
        layers.append(layer)
        idx += 1
    # return layersDict, layers
    return layers


def filterSKUByLayerFine(skulist, thres, imgInfo):
    leftIDs = getIDs(skulist)
    layerParser = layerSKU(skulist, thres)
    layersDict = {}
    layers = []
    idx = 1
    left_skulist = skulist.copy()

    while len(leftIDs) > 0:
        layerParser.getSkuAvgWH()
        layerParser.getLayerMaxAndMinYBox()

        top_skulist_temp = layerParser.skufilterCurrentLayer(left_skulist)
        top_skulist, left_skulist, leftIDs, topIDs = layerParser.skufilterFineTilt(top_skulist_temp)
        layer = Layer(imgInfo)
        layer.getBottomLine(top_skulist)
        layersDict[idx] = layer
        layers.append(layer)
        idx += 1
    # return layersDict, layers
    return layers


def leastsquaremethod(pointlist):
    if len(pointlist) < 2:
        return 0, 0
    else:
        sumx = 0
        sumy = 0
        sumxx = 0
        sumxy = 0
    for point in pointlist:
        sumx += point[0]
        sumy += point[1]
        sumxy += point[0] * point[1]
        sumxx += point[0] ** 2
    n = len(pointlist)
    ave_x = sumx / n
    ave_y = sumy / n
    if (sumxx - sumx ** 2 / n) != 0:
        k = (sumxy - sumx * sumy / n) / (sumxx - sumx ** 2 / n)
    else:
        k = 0
    b = ave_y - ave_x * k
    return k, b


def getIDs(skus):
    ids = []
    if isinstance(skus, list):
        for sku in skus:
            ids.append(sku['id'])
    if isinstance(skus, dict):
        for k, sku in skus.items():
            ids.append(sku['id'])

    return ids


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


class layerSKU:
    def __init__(self, skus, thres):
        self.leftSku = skus.copy()
        self.topCentery = 0
        self.bottomCentery = 0
        # self.thres1 = float(thres['layerThr'])
        self.thres1 = float(thres['Default'])
        self.avgH = 0
        self.avgW = 0

    def getLayerMaxAndMinYBox(self):
        """ Something wrong with """
        topSkuMaxy = 100000
        botSkuMaxy = 0
        for sku in self.leftSku:
            y = float(sku['skuMaxy'])
            centerY = sku['centerY']
            if y < topSkuMaxy:
                topSkuMaxy = y
                self.topCentery = centerY

            if y > botSkuMaxy:
                botSkuMaxy = y
                self.bottomCentery = centerY

    def skufilter(self):
        topList = []
        leftList = []
        topIDs = []
        leftIDs = []

        for sku in self.leftSku:
            centerY = sku['centerY']
            key = sku['id']
            layerdist = self.avgH * self.thres1
            if abs(centerY - self.topCentery) <= layerdist:
                topList.append(sku)
                topIDs.append(key)
            else:
                leftList.append(sku)
                leftIDs.append(key)
        self.leftSku = leftList
        # print('left sku:{} leftID:{}'.format(len(list(self.leftSkudict)),len(self.leftIDs)))
        return topList, leftList, leftIDs, topIDs

    def getSkuAvgWH(self):
        w_sum = 0
        h_sum = 0
        layerNum = 0
        for sku in self.leftSku:
            layerNum += 1
            w_sum += (float(sku['skuW']))
            h_sum += (float(sku['skuH']))

        if layerNum > 0:
            self.avgW = w_sum / layerNum
            self.avgH = h_sum / layerNum
        else:
            self.avgW = 0
            self.avgH = 0

    def skufilterCurrentLayer(self, leftSku):
        topList = []
        leftList = []
        topIDs = []
        leftIDs = []

        for sku in leftSku:
            centerY = sku['centerY']
            key = sku['id']
            layerdist = self.avgH * self.thres1
            if abs(centerY - self.topCentery) <= layerdist:
                topList.append(sku)
                topIDs.append(key)
            else:
                leftList.append(sku)
                leftIDs.append(key)
        # print('left sku:{} leftID:{}'.format(len(list(self.leftSkudict)),len(self.leftIDs)))
        return topList

    def skufilterFineTilt(self, topSkuList):
        """Based on the topSkuList maxY to get the fine layer with tilt.
        Args:
            topSkuList: raw sku of this layers.
        Returns:

        """
        topList = []
        leftList = []
        topIDs = []
        leftIDs = []

        selectRatio = 0.5
        selectNum = 1
        if (len(topSkuList)) > 1:
            selectNum = int(selectRatio * len(topSkuList))

        topSkuListSort = sorted(topSkuList, key=operator.itemgetter('skuMaxy'), reverse=True)
        bottomSku = topSkuListSort[0]

        # Get tempTopList contains all the skus.
        tempTopList = []
        avgMaxy = 0.0
        for i in range(selectNum):
            avgMaxy += float(topSkuListSort[i]['skuMaxy'])

        avgMaxy = avgMaxy / selectNum

        avgMaxy = topSkuListSort[0]['skuMaxy']
        threshold = 2.5

        for sku in self.leftSku:
            maxY = sku['skuMaxy']
            key = sku['id']
            layerdist = self.avgH * threshold

            if maxY - avgMaxy <= layerdist:
                tempTopList.append(sku)

        # Sort the tempTopList based on the skuMaxx
        tempTopListSort = sorted(tempTopList, key=operator.itemgetter('skuMaxx'))

        # Based on the bottomSku to loop at two direction.
        bottomSkuId = tempTopListSort.index(bottomSku)

        # loop to right:
        currentSku = bottomSku
        thresLayer = 0.7  # thres to judge if the sku is the same layer
        thresSku = 0.7  # thres to judge if the sku should be replace the current sku.
        for i in range(bottomSkuId, len(tempTopListSort)):
            # key=sku['id']
            key = tempTopListSort[i]['id']
            skuDiff = tempTopListSort[i]['skuMaxy'] - currentSku['skuMaxy']
            skuDistLayer = self.avgH * thresLayer
            skuDistSku = self.avgH * thresSku

            if skuDiff < skuDistLayer:
                # key=tempTopListSort[i]['id']
                topList.append(tempTopListSort[i])
                topIDs.append(key)

                if skuDiff > -1 * skuDistSku:
                    currentSku = tempTopListSort[i]
            else:
                pass

        # loop to left:
        currentSku = bottomSku
        thresLayer = 0.7  # thres to judge if the sku is the same layer
        thresSku = 0.7  # thres to judge if the sku should be replace the current sku.

        for i in range(bottomSkuId - 1, -1, -1):
            # key=sku['id']
            key = tempTopListSort[i]['id']
            skuDiff = tempTopListSort[i]['skuMaxy'] - currentSku['skuMaxy']
            skuDistLayer = self.avgH * thresLayer
            skuDistSku = self.avgH * thresSku

            if skuDiff < skuDistLayer:
                topList.append(tempTopListSort[i])
                topIDs.append(key)

                if skuDiff > -1 * skuDistSku:
                    currentSku = tempTopListSort[i]
            else:
                pass

        for sku in self.leftSku:
            if sku in topList:
                pass
            else:
                key = sku['id']
                leftList.append(sku)
                leftIDs.append(key)

        self.leftSku = leftList
        # print('left sku:{} leftID:{}'.format(len(list(self.leftSkudict)),len(self.leftIDs)))
        return topList, leftList, leftIDs, topIDs


class Layer:
    def __init__(self, imgInfo):
        self.layerskus = []
        self.skuNum = 0
        self.imgInfo = imgInfo
        self.bottom_startX = 0
        self.bottom_startY = 0
        self.bottom_endX = 0
        self.bottom_endY = 0
        self.skutop_hightest = 0
        self.skubot_lowest = 0
        self.bottom_k = 0
        self.bottom_b = 0
        self.skutop_x = 0
        self.skutop_y = 0
        self.layerAvgskuH = 0
        self.layerH = 0
        self.score = 0

        # new adding
        self.layerAvgCenterY = 0

    def getLayerskus(self):
        return self.layerskus

    # get bottom line on one layer
    def getBottomLine(self, layerskus):
        self.layerskus = layerskus
        skus = []
        sku_top = (10000, 10000)
        sku_bot = 0
        sku_hsum = 0
        sku_centerysum = 0
        # pdb.set_trace()
        for sku in layerskus:
            self.skuNum += 1
            x_min = sku['skuMinx']
            x_max = sku['skuMaxx']
            y_min = sku['skuMiny']
            y_max = sku['skuMaxy']
            # new adding
            centery = sku['centerY']
            sku_centerysum += float(centery)
            sku_hsum = abs(y_max - y_min)
            if sku_top[1] > y_min:
                sku_top = x_min, y_min
            if sku_bot < y_max:
                sku_bot = y_max
            skus.append((x_min, y_max))
            skus.append((x_max, y_max))
        # new adding
        self.layerAvgCenterY = sku_centerysum / self.skuNum
        self.layerAvgskuH = int(float(sku_hsum) / self.skuNum)
        self.layerH = self.layerAvgskuH * 1.2
        self.skutop_x, self.skutop_y = sku_top
        self.skubot_lowest = sku_bot
        botparam = leastsquaremethod(skus)
        self.bottom_k = botparam[0]
        self.bottom_b = botparam[1]
        self.bottom_startX = 0
        self.bottom_startY = botparam[1]
        self.bottom_endX = int(self.imgInfo["width"])
        self.bottom_endY = int(self.imgInfo["width"]) * botparam[0] + botparam[1]

    def getline(self):
        return {'xmin': self.bottom_startX, 'ymin': self.bottom_startY, 'xmax': self.bottom_endX,
                'ymax': self.bottom_endY}


class ConvertLayers():
    def __init__(self, imgInfo, layers):
        self.imgInfo = imgInfo
        self.width = int(imgInfo["width"])
        self.height = int(imgInfo["height"])
        self.layers = layers

    def sortByB(self, elem):
        return elem[1]

    def sortBySkuname(self, elem):
        return elem["skuName"]

    def convert(self):

        if not self.layers:
            return {
                'layerLine': [],
                'layerNum': 0,
                'boxInfo': [],
                'imageInfo': {},
                'skuStat': [],
                'layerOut': []
            }

        laylist = []
        linelist = []
        skuStat = {}
        # self.layers.sort(key=lambda k:k.bottom_startY)
        self.layers.sort(key=lambda k: k.layerAvgCenterY)
        rate = [0.0]
        for idx, layer in enumerate(self.layers):
            line = layer.getline()
            skulist = []
            linelist.append(line)
            for sku in layer.layerskus:
                skulist.append(sku)
            skulist.sort(key=lambda k: k['centerX'])
            for sku in skulist:
                laylist.append(
                    {
                        'layer': str(idx + 1),
                        'skuName': sku['skuName'],
                        'location': sku['location'],
                        'score': round(float(sku['score']), 2) * 100
                    }
                )

        # sku layerOut detect
        layerOut = {}
        for idx, layer in enumerate(self.layers):
            for sku in layer.layerskus:
                layerNum = str(idx + 1)
                if sku['skuName'] in layerOut:
                    if layerNum in layerOut[sku['skuName']]:
                        skuStack = layerOut[sku['skuName']][layerNum]
                        if len(sku['layerOut']) + 1 > skuStack:
                            layerOut[sku['skuName']][layerNum] = len(sku['layerOut'])
                    else:
                        layerOut[sku['skuName']][layerNum] = len(sku['layerOut'])

                elif sku['skuName'] != 'daiding_101':
                    layerOut[sku['skuName']] = {}
                    layerOut[sku['skuName']][layerNum] = len(sku['layerOut'])

        finalLayerout = []
        for name, layers in layerOut.items():
            lay = {'skuName': name}
            count = []
            for layername, out in layers.items():
                count.append(out)
            lay['count'] = sum(count)

            finalLayerout.append(lay)

        stat = []
        layNum = len(linelist)
        return {
            'layerLine': linelist,
            'layerNum': layNum,
            'boxInfo': laylist,
            'imageInfo': self.imgInfo,
            'skuStat': stat,
            'layerOut': finalLayerout
        }


class SKU:
    def __init__(self, skulist, imgInfo):
        self.skulist = skulist
        self.encodedSkus = {}
        self.allIDs = []
        self.leftSkudict = {}
        self.precision = 3
        self.imgW = imgInfo['width']
        self.imgH = imgInfo['height']
        self.skuEncoder()
        # pdb.set_trace()
        # self.stackSKUProc()

    def skuEncoder(self):
        # encode sku
        for idx, sku in enumerate(self.skulist):
            score = sku[4]
            loc = sku[:-2]
            loc[0] = int(loc[0] * self.imgW)
            loc[1] = int(loc[1] * self.imgH)
            loc[2] = int(loc[2] * self.imgW)
            loc[3] = int(loc[3] * self.imgH)
            name = sku[-1]
            loc = list(map(int, loc))
            self.encodedSkus[idx] = {
                'skuName': name,
                'skuH': loc[3] - loc[1],
                'skuW': loc[2] - loc[0],
                'skuMinx': loc[0],
                'skuMiny': loc[1],
                'skuMaxx': loc[2],
                'skuMaxy': loc[3],
                'score': score,
                'centerX': round((loc[2] + loc[0]) * 0.5),
                'centerY': round((loc[3] + loc[1]) * 0.5),
                'sameSku': False,
                'id': idx,
                'layer': 0,
                'layerOut': [],
                'location':
                    {
                        'xmin': round(float(loc[0]) / self.imgW, self.precision),
                        'ymin': round(float(loc[1]) / self.imgH, self.precision),
                        'xmax': round(float(loc[2]) / self.imgW, self.precision),
                        'ymax': round(float(loc[3]) / self.imgH, self.precision)},
            }
            self.allIDs.append(idx)

    def getskudict(self):
        return self.encodedSkus


def getskuList(skus):
    skulist = []
    for k, skus in skus.items():
        skulist.append(skus)
    # self.skulist=skulist
    return skulist
