from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str

def get_video_data(image, pred, label):
    result_dict = {}
    new_label = []
    new_pred = []
    new_names = []
    # print(image[0])
    # print(pred.shape)
    # print(label.shape)
    for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
        # 分割字符串，获取'a'和'b'的值
        s = item[0]
        if '\\' in s:
            parts = s.split('\\')
        else:
            parts = s.split('/')
        a = parts[-2]
        b = parts[-1]

        # 如果'a'的值还没有在字典中，添加一个新的键值对
        if a not in result_dict:
            result_dict[a] = []

        # 将'b'的值添加到'a'的列表中
        result_dict[a].append(item)
    image_arr = list(result_dict.items())
    # 将字典的值转换为一个列表，得到二维数组

    #pred for video = avg pred across all frames
    #other possibilities: calculate threshold per frame?
    for video_name, video in image_arr:
        pred_sum = 0
        label_sum = 0
        leng = 0
        for frame in video:
            pred_sum += float(frame[1])
            label_sum += int(frame[2])
            leng += 1
        new_names.append(video_name)
        new_pred.append(pred_sum / leng)
        new_label.append(int(label_sum / leng))
    return (new_names, new_pred, new_label)
        

def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        _, new_pred, new_label = get_video_data(image, pred, label)
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    try:
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    except:
        eer = 0
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        try: 
            v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
        except:
            v_auc=auc
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
