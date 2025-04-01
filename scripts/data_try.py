from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
import torch

def main():
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
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=0
    )
    # 测试数据加载器
    test_data_loaders = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed
    )

    
    # 5. 使用数据加载器
    print("开始加载训练数据...")
    for iteration, data in enumerate(train_data_loader):
        # data 通常包含:
        # - images: 图片张量
        # - targets: 标注信息（边界框、关系等）
        images = data[0]  # 图片数据
        targets = data[1]  # 标注数据
        
        print(f"批次 {iteration}:")
        print(f"图片张量形状: {images.tensors.shape}")
        print(f"目标数量: {len(targets)}")
        
        # 只打印前两个批次的信息
        if iteration >= 1:
            break
            
    print("\n开始加载测试数据...")
    for test_data_loader in test_data_loaders:
        for iteration, data in enumerate(test_data_loader):
            images = data[0]
            targets = data[1]
            
            print(f"批次 {iteration}:")
            print(f"图片张量形状: {images.tensors.shape}")
            print(f"目标数量: {len(targets)}")
            
            if iteration >= 1:
                break

if __name__ == "__main__":
    main()
