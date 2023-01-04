# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import cv2 as cv


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)


    ##################################################
    ######插入计算thining后的交并区域大小
    ####先对预测的进行thinning
    # 顾加thinning算法
    A = pred_label.detach().cpu().numpy()  # tensor转换为ndarray  如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。
    A = np.array(A,np.uint8)  #不转为uint8的话会报错，因为numpy默认是32位的float浮点数而不是uint8  error：(expected: 'processed.type() == CV_8UC1'), where 'processed.type()' is 5 (CV_32FC1)
    #现在是(c,w,h), 要转为(w,h,c),但实验证明这样还是不行，直接squeeze掉维数为1的那一维
    # A = numpy.transpose(A,(1,2,0)) 
    # A=np.squeeze(A)  #就变成(512，512)了
    A=A*255  #thinning需要在0-255的范围
    # cv.imwrite('/home/gnh/code/mmsegmentation-master/test0.png',A)
    B = cv.ximgproc.thinning(A, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)  #Enforce the range of the input image to be in between 0 - 255
    # cv.imwrite('/home/gnh/code/mmsegmentation-master/test1.png',B)
    # print(B)
    B=B/255
    B = np.array(B,np.int64) 
    B = torch.from_numpy(B) # ndarray转换为tensor
    # B=B.unsqueeze(0)  #再把channel那第一维补上变成[1, 512, 512]
    pred_label_thin=B
    # pred_label_thin=pred_label_thin.cuda()
    
    ####在对gt的进行thinning
    # 顾加thinning算法
    A1 = label.detach().cpu().numpy()  # tensor转换为ndarray
    A1 = np.array(A1,np.uint8)  #不转为uint8的话会报错，因为numpy默认是32位的float浮点数而不是uint8  error：(expected: 'processed.type() == CV_8UC1'), where 'processed.type()' is 5 (CV_32FC1)
    #现在是(c,w,h), 要转为(w,h,c),但实验证明这样还是不行，直接squeeze掉维数为1的那一维
    # A = numpy.transpose(A,(1,2,0)) 
    # A=np.squeeze(A)  #就变成(512，512)了
    A1=A1*255  #thinning需要在0-255的范围
    # cv.imwrite('/home/gnh/code/mmsegmentation-master/test0.png',A)
    B1 = cv.ximgproc.thinning(A1, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)  #Enforce the range of the input image to be in between 0 - 255
    # cv.imwrite('/home/gnh/code/mmsegmentation-master/test1.png',B)
    # print(B)
    B1=B1/255
    B1 = np.array(B1,np.int64) 
    B1 = torch.from_numpy(B1) # ndarray转换为tensor
    # B=B.unsqueeze(0)  #再把channel那第一维补上变成[1, 512, 512]
    label_thin=B1
    # label_thin=label_thin.cuda()
    ##################################################

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index) #mask为512*512的布尔矩阵，用于下两行代码筛掉predict、gt为255的（实际上如果是255就说明这个标签是有问题的，我们需要的是0，1的而不是0，255的）
    pred_label = pred_label[mask]#布尔运算后，将矩阵拉直为一行
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

####################Precision Thinning part#################
####predict进行了thin，gt没有
    pred_label_thin = pred_label_thin[mask]#布尔运算后，将矩阵拉直为一行
    # label_thin = label_thin[mask]

    intersect_precision = pred_label_thin[pred_label_thin == label]
    area_intersect_precision = torch.histc(
        intersect_precision.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label_thin = torch.histc(
        pred_label_thin.float(), bins=(num_classes), min=0, max=num_classes - 1)
    # area_label = torch.histc(
    #     label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union_precision = area_pred_label_thin + area_label - area_intersect_precision
##################################################
####################Recall Thinning part#################
####gt进行了thin，predict没有
    # pred_label_thin = pred_label_thin[mask]#布尔运算后，将矩阵拉直为一行
    label_thin = label_thin[mask]

    intersect_recall = pred_label[pred_label == label_thin]
    area_intersect_recall = torch.histc(
        intersect_recall.float(), bins=(num_classes), min=0, max=num_classes - 1)
    # area_pred_label_thin = torch.histc(
    #     pred_label_thin.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label_thin = torch.histc(
        label_thin.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union_recall = area_pred_label + area_label_thin - area_intersect_recall
##################################################

    return area_intersect, area_union, area_pred_label, area_label,\
    area_intersect_precision, area_union_precision, area_pred_label_thin,\
    area_intersect_recall, area_union_recall, area_label_thin


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label   


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 10 or 4 #原本为4 因为加了6个额外的输出来计算新的指标，我修改为10

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    if(len(pre_eval_results) == 10):
        total_area_intersect_precision = sum(pre_eval_results[4])
        total_area_union_precision = sum(pre_eval_results[5])
        total_area_pred_label_thin = sum(pre_eval_results[6])

        total_area_intersect_recall = sum(pre_eval_results[7])
        total_area_union_recall = sum(pre_eval_results[8])
        total_area_label_thin = sum(pre_eval_results[9])
    
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label, total_area_label,
                                        total_area_intersect_precision, total_area_union_precision, total_area_pred_label_thin, 
                                        total_area_intersect_recall, total_area_union_recall, total_area_label_thin,
                                        metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          total_area_intersect_precision, total_area_union_precision, total_area_pred_label_thin, 
                          total_area_intersect_recall, total_area_union_recall, total_area_label_thin,            
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore','newmFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        #################newmFscore########################################
        elif metric == 'newmFscore':                 
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

            f_value_np=f_value.numpy()
            f_value_list=list(f_value_np)
            f_value_str = " ".join(str(x) for x in f_value_list)
            newfscore_path="/home/geh/code/mmsegmentation-master/work_dirs/GAPs_unet_weightdecay/test2.txt"
            with open(newfscore_path,"a+") as f:
                f.write(f_value_str+' ')  # 

            newprecision = total_area_intersect_precision / total_area_pred_label_thin
            newrecall = total_area_intersect_recall / total_area_label_thin
            newf_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(newprecision, newrecall)])
            ret_metrics['newFscore'] = newf_value
            ret_metrics['newPrecision'] = newprecision
            ret_metrics['newRecall'] = newrecall

            newf_value_np=newf_value.numpy()
            newf_value_list=list(newf_value_np)
            newf_value_str = " ".join(str(x) for x in newf_value_list)
            with open(newfscore_path,"a+") as f:
                f.write(newf_value_str+'\n')  # 
        ###################################################################
    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
