import os
import sys
import torch

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from sgg_benchmark.config import cfg
from sgg_benchmark.data.datasets import VGDataset
from sgg_benchmark.data.datasets.evaluation import evaluate
from sgg_benchmark.utils.logger import setup_logger

def test_vg_evaluation():
    # 1. 设置配置    
    # cfg.MODEL.BACKBONE.CONV_BODY = "R-101-FPN"  # 设置backbone
    # cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 151  # VG数据集的类别数
    # cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51  # VG数据集的关系类别数
    
    # cfg.merge_from_file("configs/VrR-VG/e2e_relation_detector_X_101_32_8_FPN_1x.yaml")  # 使用预设配置文件
    # cfg.freeze()

    # 2. 设置输出目录和logger
    output_dir = "output/vg_evaluation_test"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("sgg_benchmark", output_dir)

    # 3. 创建数据集实例
    dataset = VGDataset(
        split="all",
        img_dir="datasets/vg/VG_100K",
        roidb_file="datasets/vg/VG-SGG.h5",
        dict_file="datasets/vg/VG-SGG-dicts.json",
        image_file="datasets/vg/image_data.json",
    )

    # 4. 创建模拟的预测结果
    # 这里创建一个简单的预测结果示例
    def create_mock_prediction():
        # 创建一个示例预测结果
        boxes = torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]], dtype=torch.float32)
        labels = torch.tensor([1, 2])  # 假设类别标签
        scores = torch.tensor([0.9, 0.8])
        
        # 关系预测
        rel_pairs = torch.tensor([[0, 1]])  # 表示第0个物体和第1个物体之间有关系
        rel_labels = torch.tensor([1])  # 关系类别
        rel_scores = torch.tensor([0.7])

        prediction = {
            "bbox": boxes,
            "labels": labels,
            "scores": scores,
            "rel_pair_idxs": rel_pairs,
            "pred_rel_labels": rel_labels,
            "rel_scores": rel_scores,
        }
        return prediction

    # 为每张图片创建预测结果
    predictions = [create_mock_prediction() for _ in range(len(dataset))]

    # 5. 执行评估
    results = evaluate(
        cfg=cfg,
        dataset=dataset,
        dataset_name="vg_test",
        predictions=predictions,
        output_folder=output_dir,
        logger=logger,
        iou_types=["bbox", "relations"],
        informative=True
    )

    # 6. 打印结果
    logger.info("Evaluation Results:")
    logger.info(results)

if __name__ == "__main__":
    test_vg_evaluation()