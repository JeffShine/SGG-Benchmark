import torch
import os
import json
import logging
from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.data.datasets.evaluation.vg.vg_eval_custom import do_vg_evaluation

# 设置日志器
def setup_logger():
    logger = logging.getLogger("scene_graph_evaluation")
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(ch)
    
    return logger

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

def evaluate_scene_graph_custom():
    # 创建日志器
    logger = setup_logger()
    
    # 创建数据集
    dataset = DummyDataset()
    
    # 创建predictions
    predictions = [create_sample_prediction() for _ in range(2)]  # 创建2张图片的预测结果
    
    # 创建输出目录
    output_folder = "output/dummy_evaluation_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建初始结果文件
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump({}, f)
    
    # 设置评估参数
    iou_types = ("bbox", "relations")
    
    # 使用修改后的do_vg_evaluation函数进行评估
    result_dict = do_vg_evaluation(
        dataset=dataset,
        dataset_name="VG_stanford_filtered_with_attribute_test",  # 数据集名称
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
        test_informative=False,
        use_gt_box=False,
        use_gt_object_label=False,
        num_rel_classes=5,  # 关系类别数量
        multiple_preds=False,
        iou_threshold=0.5,
        top_k=100,
        informative=False
    )
    
    # 打印评估结果
    print("\n评估结果:")
    print("="*50)
    if 'sgdet_recall' in result_dict:
        print("场景图检测召回率@K:")
        for k in [20, 50, 100]:
            print(f"R@{k}: {result_dict['sgdet_recall'][k]}")
    
    if 'sgdet_mean_recall' in result_dict:
        print("\n场景图检测平均召回率@K:")
        for k in [20, 50, 100]:
            print(f"mR@{k}: {result_dict['sgdet_mean_recall'][k]}")
            
    if 'mAP' in result_dict:
        print(f"\n检测mAP: {result_dict['mAP']}")

if __name__ == "__main__":
    evaluate_scene_graph_custom()