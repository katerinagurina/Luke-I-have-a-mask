#by Andrei Erofeev
import tqdm
import torch
import numpy as np

def single_IoU(true_bbox, pred_bbox, out_type = 'xy'):
    if out_type == 'xy':
        x_left = max(true_bbox[0], pred_bbox[0])
        y_top = max(true_bbox[1], pred_bbox[1])
        x_right = min(true_bbox[2], pred_bbox[2])
        y_bottom = min(true_bbox[3], pred_bbox[3])
    elif out_type == 'wh':
        x_left = max(true_bbox[0], pred_bbox[0])
        y_top = max(true_bbox[1], pred_bbox[1])
        x_right = min(true_bbox[0] + true_bbox[2], pred_bbox[2])
        y_bottom = min(true_bbox[1] + true_bbox[3], pred_bbox[3])
    else:
        raise AttributeError('Unknown true bbox input type, choose between "wh" and "xy"')

    if x_right < x_left or y_bottom < y_top:
        return 0

    inters_area = (x_right - x_left) * (y_bottom - y_top)

    if out_type == 'xy':
        true_bbox_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    elif out_type == 'wh':
        true_bbox_area = true_bbox[2] * true_bbox[3]
    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    iou = inters_area / float(true_bbox_area + pred_bbox_area - inters_area)

    assert iou.item() >= 0.0 or iou.item() <= 1.0

    return iou


def mAP(true_dataset, pred_dataset, out_type = 'xy', iou_thresh=0.5):
    '''
    Function for calculation mean average precision and average IoU over all dataset.
    :param true_dataset: dataset instance, which items contains true bboxes in 1-st elemnt
    :param pred_dataset: array-like (or list) of lists with predicted bbboxes
    :param out_type: type of true bboxes format - whether 3d and 4th elements represent coordinates or width and height,
    could be xy or wh
    :param iou_thresh: threshold for IoU calculation
    :return: dectionary with two values: average IoU and mean Average Precision
    '''

    assert out_type == 'xy' or out_type == 'wh'

    len_data = len(true_dataset)
    data_total_iou = 0
    data_total_precision = 0
    for i in tqdm.tqdm(range(len_data)):
        true_bboxes = true_dataset[i][1]
        pred_boxes = pred_dataset[i]

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        total_gt_face = len(true_bboxes)
        for gt in true_bboxes:
            max_iou_per_gt = 0
            for i, pred in enumerate(pred_boxes):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = single_IoU(gt, pred, out_type)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    if pred_dict[i] >= iou_thresh:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

    result = dict()
    result['average_iou'] = float(data_total_iou) / float(len_data)
    result['mean_average_precision'] = float(data_total_precision) / float(len_data)
    return result


def multilabel_mAP(true_dataset, pred_dataset, pred_labels=None, true_labels=None, num_classes=3, out_type='xy',
                   iou_thresh=0.5):
    '''
    Function for calculation multilabel mean average precision and average IoU over all dataset.
    :param true_dataset: dataset instance, which items contains true bboxes in 1-st elemnt
    :param pred_dataset: array-like (or list) of lists with predicted bbboxes
    :param out_type: type of true bboxes format - whether 3d and 4th elements represent coordinates or width and height,
    could be xy or wh
    :param iou_thresh: threshold for IoU calculation
    :return: dectionary with two values: average IoU and mean Average Precision
    '''

    assert out_type == 'xy' or out_type == 'wh'

    if true_labels is None:
        true_labels = [true_dataset[i][1][:, -1].cpu().detach().numpy() for i in range(len(true_dataset))]

    len_data = len(true_dataset)
    average_ious = []
    aps = []
    for i in range(num_classes):

        data_label_iou = 0
        data_label_precision = 0
        for j in tqdm.tqdm(range(len_data)):
            true_bboxes = true_dataset[j][1]
            pred_bboxes = pred_dataset[j]
            if len(pred_bboxes) > 0:
                bool_ind = pred_labels[j] == i
                ind = np.array(range(len(bool_ind)))
                ind = ind[bool_ind]
                pred_bboxes = [pred_bboxes[k] for k in ind]
                # pred_bboxes = [pred_bboxes[int(ind)]]

            bool_ind = true_labels[j] == i
            ind = np.array(range(len(bool_ind)))
            ind = ind[bool_ind]
            true_bboxes = true_bboxes[ind]

            label_iou = 0
            tp = 0
            pred_dict = dict()
            label_gt_face = len(true_bboxes)

            for gt in true_bboxes:
                max_iou_per_gt = 0
                for i, pred in enumerate(pred_bboxes):
                    if i not in pred_dict.keys():
                        pred_dict[i] = 0
                    iou = single_IoU(gt, pred, out_type)
                    if iou > max_iou_per_gt:
                        max_iou_per_gt = iou
                    if iou > pred_dict[i]:
                        pred_dict[i] = iou
                label_iou = label_iou + max_iou_per_gt

            if label_gt_face != 0:
                if len(pred_dict.keys()) > 0:
                    for i in pred_dict:
                        if pred_dict[i] >= iou_thresh:
                            tp += 1
                    precision = float(tp) / float(label_gt_face)

                else:
                    precision = 0

                image_average_iou = label_iou / label_gt_face
                image_average_precision = precision

                data_label_iou += image_average_iou
                data_label_precision += image_average_precision
        result = dict()
        average_ious.append(float(data_label_iou) / float(len_data))
        aps.append(float(data_label_precision) / float(len_data))
    result = dict()
    result['average_iou'] = np.mean(average_ious)
    result['mean_average_precision'] = np.mean(aps)
    return result