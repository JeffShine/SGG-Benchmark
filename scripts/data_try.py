from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
import torch
from sgg_benchmark.data.datasets.evaluation.vg.vg_eval_custom import do_vg_evaluation
import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.data.datasets.evaluation.vg.sgg_eval import *
from sgg_benchmark.config.paths_catalog import DatasetCatalog

def eval_vg(predictions):
    # 1. 配置基本参数
    cfg.merge_from_file("configs/VG150/e2e_relation_X_101_32_8_FPN_1x.yaml")  # 需要指定具体的配置文件路径
    # cfg.merge_from_file("configs/PSG/e2e_relation_detector_X_101_32_8_FPN_1x.yaml")  # 需要指定具体的配置文件路径
    # 2. 设置一些必要的配置
    # cfg.SOLVER.IMS_PER_BATCH = 2  # 每批次图片数
    # cfg.DATALOADER.NUM_WORKERS = 2  # 数据加载的工作进程数
    # cfg.DATASETS.TRAIN = ("vg_train",)  # 训练集名称
    # cfg.DATASETS.TEST = ("vg_test",)    # 测试集名称
    
    # 3. 初始化分布式训练设置
    distributed = False  # 是否使用分布式训练
    
    # 4. 创建数据加载器
    # 训练数据加载器
    # train_data_loader = make_data_loader(
    #     cfg,
    #     mode='train',
    #     is_distributed=distributed,
    #     start_iter=0
    # )
    # train_dataset = train_data_loader.dataset
    # 测试数据加载器
    test_data_loaders = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed
    )
    test_dataset = test_data_loaders[0].dataset
    logger = logging.getLogger("sgg_benchmark.eval_try")

    return do_vg_evaluation(
        test_dataset,
        dataset_name='VG150',
        predictions=predictions,
        output_folder='output/VG150_try',
        logger=logger,
        iou_types=('bbox',),
        test_informative=False,
        use_gt_box=False,
        use_gt_object_label=False,
        num_rel_classes=50,
        multiple_preds=False,
        iou_threshold=0.5,
        top_k=100,
        informative=False,
    )
    # # 5. 使用数据加载器
    # print("开始加载训练数据...")
    # for iteration, data in enumerate(train_data_loader):
    #     # data 通常包含:
    #     # - images: 图片张量
    #     # - targets: 标注信息（边界框、关系等）
    #     images = data[0]  # 图片数据
    #     targets = data[1]  # 标注数据
        
    #     print(f"批次 {iteration}:")
    #     print(f"图片张量形状: {images.tensors.shape}")
    #     print(f"目标数量: {len(targets)}")
    #     print(f"targets: {targets}")
        
    #     # 只打印前两个批次的信息
    #     if iteration >= 1:
    #         break
            
    # print("\n开始加载测试数据...")
    # for test_data_loader in test_data_loaders:
    #     for iteration, data in enumerate(test_data_loader):
    #         images = data[0]
    #         targets = data[1]
            
    #         print(f"批次 {iteration}:")
    #         print(f"图片张量形状: {images.tensors.shape}")
    #         print(f"目标数量: {len(targets)}")
    #         print(f"targets: {targets}")
    #         import pdb; pdb.set_trace()
    #         if iteration >= 1:
    #             break



class DummyDataset:
    """模拟数据集类"""
    def __init__(self):
        self.ind_to_classes = ['__background__', 'person', 'chair', 'table']  # 示例类别
        self.ind_to_predicates = ['__background__', 'sitting_on', 'near', 'at', 'on']  # 示例关系
        
    def get_img_info(self, index):
        return {"width": 800, "height": 600}
        
    def get_groundtruth(self, index, evaluation=False):
        # 创建groundtruth示例
        gt_boxes = torch.tensor([
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
            [500.0, 100.0, 600.0, 200.0],
        ])
        
        gt = BoxList(gt_boxes, (800, 600), mode='xyxy')
        
        # 添加标签
        gt_labels = torch.tensor([1, 2, 3]).long()  # person, chair, table
        gt.add_field('labels', gt_labels)
        
        # 添加关系三元组 (subject_idx, object_idx, predicate_label)
        gt_rels = torch.tensor([
            [0, 1, 1],  # person sitting_on chair
            [1, 2, 3],  # chair at table
        ]).long()
        gt.add_field('relation_tuple', gt_rels)
        
        return gt
    
    def get_statistics(self):
        # 添加需要的统计信息
        stats = {}
        stats['rel_classes'] = self.ind_to_predicates
        stats['obj_classes'] = self.ind_to_classes
        
        # 创建一个简单的前景关系矩阵
        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        fg_matrix = torch.zeros((num_obj_classes, num_obj_classes, num_rel_classes))
        
        # 添加一些有效关系
        fg_matrix[1, 2, 1] = 1  # person-chair-sitting_on
        fg_matrix[2, 3, 3] = 1  # chair-table-at
        
        stats['fg_matrix'] = fg_matrix
        
        return stats

def create_sample_prediction(image_width=800, image_height=600):
    """创建一个样例prediction"""
    pred_boxes = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],
        [300.0, 300.0, 400.0, 400.0],
        [500.0, 100.0, 600.0, 200.0],
    ])
    
    prediction = BoxList(pred_boxes, (image_width, image_height), mode='xyxy')
    
    # 对象检测分数
    pred_scores = torch.tensor([0.9, 0.85, 0.95])
    prediction.add_field('pred_scores', pred_scores)
    
    # 对象类别标签
    pred_labels = torch.tensor([1, 2, 3]).long()
    prediction.add_field('pred_labels', pred_labels)
    
    # 关系对索引
    rel_pair_idxs = torch.tensor([
        [0, 1],
        [1, 2],
    ]).long()
    prediction.add_field('rel_pair_idxs', rel_pair_idxs)
    
    # 关系预测分数
    pred_rel_scores = torch.zeros((2, 5))  # 5个关系类别（包括背景）
    pred_rel_scores[0] = torch.tensor([0.1, 0.7, 0.1, 0.05, 0.05])
    pred_rel_scores[1] = torch.tensor([0.05, 0.05, 0.1, 0.7, 0.1])
    prediction.add_field('pred_rel_scores', pred_rel_scores)
    
    # 关系标签
    pred_rel_labels = torch.tensor([1, 3]).long()
    prediction.add_field('pred_rel_labels', pred_rel_labels)
    
    return prediction
if __name__ == "__main__":
    predictions = create_sample_prediction()
    import pdb; pdb.set_trace()
    eval_vg(predictions=predictions)
