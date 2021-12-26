def merge_sku_others(skulist, otherslist):
    if len(otherslist) == 0:
        return skulist
    if len(skulist) == 0:
        return otherslist
    del_list = []

    for i in range(len(skulist)):
        for j in range(len(otherslist)):
            iou = get_iou(skulist[i], otherslist[j])
            if iou > 0.7:
                del_list.append(j)
    del_list = list(set(del_list))
    del_list.sort(reverse=True)
    for i in del_list:
        del otherslist[i]
    skulist.extend(otherslist)
    return skulist


def between_class_nms_by_score(skulist, excludelist=[], threshold=0.7):
    length = len(skulist)
    if length > 0:
        del_list = []
        for i in range(length):
            for j in range(i + 1, length):
                if skulist[i][5] not in excludelist and skulist[j][5] not in excludelist:
                    iou = get_iou(skulist[i], skulist[j])
                    if iou > threshold:
                        if skulist[i][4] < skulist[j][4]:
                            del_list.append(i)
                        else:
                            del_list.append(j)
        del_list = list(set(del_list))
        del_list.sort(reverse=True)
        for i in del_list:
            del skulist[i]
        return skulist
    else:
        return []


def between_class_nms_by_score_union(skulist, excludelist=[], threshold=0.7):
    length = len(skulist)
    if length > 0:
        del_list = []
        for i in range(length):
            for j in range(i + 1, length):
                if skulist[i][5] not in excludelist and skulist[j][5] not in excludelist:
                    iou = get_iou_Union(skulist[i], skulist[j])
                    if iou > threshold:
                        if skulist[i][4] < skulist[j][4]:
                            del_list.append(i)
                        else:
                            del_list.append(j)
        del_list = list(set(del_list))
        del_list.sort(reverse=True)
        for i in del_list:
            del skulist[i]
        return skulist
    else:
        return []


def between_class_nms_by_MinArea(skulist, excludelist=[], threshold=0.8):
    length = len(skulist)
    if length > 0:
        del_list = []
        for i in range(length):
            for j in range(i + 1, length):
                if skulist[i][5] not in excludelist and skulist[j][5] not in excludelist:
                    area_i = get_area(skulist[i])
                    area_j = get_area(skulist[j])
                    iou = get_iou(skulist[i], skulist[j])
                    if iou > threshold:
                        if area_i < area_j:
                            del_list.append(i)
                        else:
                            del_list.append(j)
        del_list = list(set(del_list))
        del_list.sort(reverse=True)
        for i in del_list:
            del skulist[i]
        return skulist
    else:
        return []


def get_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_iou(sku1, sku2):
    xmin = max(sku1[0], sku2[0])
    xmax = min(sku1[2], sku2[2])
    ymin = max(sku1[1], sku2[1])
    ymax = min(sku1[3], sku2[3])
    iou_area = 0
    if xmin <= xmax and ymin <= ymax:
        iou_area = (xmax - xmin) * (ymax - ymin)
    area1 = get_area(sku1)
    area2 = get_area(sku2)
    iou = iou_area / min(area1, area2)
    return iou


def get_iou_Union(sku1, sku2):
    xmin = max(sku1[0], sku2[0])
    xmax = min(sku1[2], sku2[2])
    ymin = max(sku1[1], sku2[1])
    ymax = min(sku1[3], sku2[3])
    iou_area = 0
    if xmin <= xmax and ymin <= ymax:
        iou_area = (xmax - xmin) * (ymax - ymin)
    area1 = get_area(sku1)
    area2 = get_area(sku2)
    iou = iou_area / (area1 + area2 - iou_area)
    return iou

def delete_special_sku(skulist, conditionlist=["fake", "BX"]):
    if len(conditionlist) == 0:
        return skulist
    del_list = []
    for i in range(len(skulist)):
        for j in range(len(conditionlist)):
            if conditionlist[j] in skulist[i][5]:
                del_list.append(i)
    del_list = list(set(del_list))
    del_list.sort(reverse=True)
    for i in del_list:
        del skulist[i]
    return skulist
    tempDict = skus.copy()
