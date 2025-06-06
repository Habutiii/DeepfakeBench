"""
eval pretained model.
"""
import os
import sys
import numpy as np
from os.path import join
import cv2
import csv
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from .dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.abstract_dataset_resize import AbstractResizeBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.utils import get_test_metrics, get_video_data
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

on_2060 = False #"2060" in torch.cuda.get_device_name()
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = AbstractResizeBaseDataset(
                config=config,
                size=256
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())

        #if i == 4:
            #break
    
    return np.array(prediction_lists), np.array(label_lists) #,np.array(feature_lists)
    
def test_epoch(model, test_data_loaders, model_name):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        
        # compute loss for each dataset
        start_time = time.time()
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        #metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              #img_names=data_dict['image'][:5*32])
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        output = ""

        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")
            output += f"{k}: {v}" + '\n'

        output += f"time: {time.time() - start_time}"

        #write_to_csv(key, data_dict['image'][:5*32], predictions_nps, label_nps)
        output_path = Path(__file__).parent.parent / f"results/{model_name}"
        os.makedirs(output_path, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{key}_{ts}"
        write_to_txt(output_path / filename, output)
        write_to_csv(output_path / filename, data_dict['image'], predictions_nps, label_nps)
        if type(data_dict['image'][0]) is not list:
            video_names, video_preds, video_labels = get_video_data(data_dict['image'], predictions_nps, label_nps)
            write_to_csv(output_path / f'{model_name}_{key}_video_{ts}', video_names, video_preds, video_labels)

    return metrics_all_datasets

def write_to_txt(name, output):
    txt_name = f'{name}_results.txt'
    with open(txt_name, 'w') as file:
        file.write(output)

def write_to_csv(name, img_names, y_pred, y_true):
    csv_name = f'{name}_results.csv'
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['file', 'predicted', 'label'])
        
        for img_name, pred, label in zip(img_names, y_pred, y_true):
            writer.writerow([img_name, pred, label]) 
    

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main(detector_path, test_datasets=None, weights_path=None):
    # parse options and load config
    
    config = {}
        
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
        
    
    config['workers'] = 0
    
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
    else:
        #config['workers'] = 8
        config['lmdb_dir'] = r'./data/LMDBs'

    # If arguments are provided, they will overwrite the yaml settings
    if test_datasets:
        config['test_dataset'] = test_datasets
    if weights_path:
        config['weights_path'] = weights_path
        config['pretrained'] = weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    print("weights path: ", weights_path)
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
            
        try:
            ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(ckpt, strict=True)
            print('===> Load checkpoint done!')
        except Exception as e:
            print("Unable to load checkpoint: it may be loaded when building the backbone")
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, config['model_name'])
    print('===> Test Done!')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.detector_path, args.test_dataset, args.weights_path)