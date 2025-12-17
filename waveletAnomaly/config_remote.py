# config_remote.py 中的关键修改
import torch

class Config:
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    SEED = 555
    
    # 数据集路径配置
    # TRAIN_ROOT_DIR = '../../dataset/UCSD/UCSDped2/Train'
    # TEST_ROOT_DIR = '../../dataset/UCSD/UCSDped2/Test'
    # LABELS_DIR = '../../dataset/UCSD/UCSDped2/label'
    TRAIN_ROOT_DIR = 'E:/dataset/UCSD/UCSDped2/Train'
    TEST_ROOT_DIR = 'E:/dataset/UCSD/UCSDped2/Test'
    LABELS_DIR = 'E:/dataset/UCSD/UCSDped2/label'


    # 数据集分段配置
    DATASET_PARAMS = {
        'segment_length': 8,      # 每个视频段的帧数
        'overlap_ratio': 0.5       # 视频段重叠比例
    }

    # 数据预处理配置
    IMAGE_SIZE = (128, 128)  # 降低图像尺寸以减少计算量
    FILE_TYPE = '.tif'
    MAX_FRAMES = 100  # 减少最大帧数以加快训练
    
    # 模型配置
    MODEL_PARAMS = {
        'in_channels': 1,
        'hidden_channels': 64,   # 增加隐藏通道数以提高表达能力
        'out_channels': 32,      # 增加输出通道数以提高表达能力
        'grid_size': 8,
        'temporal_window': 4
    }
    
    # 小波变换配置
    WAVELET_PARAMS = {
        'wavelet': 'haar',
        'levels': 2
    }
    
    # 训练配置
    TRAINING_PARAMS = {
        'batch_size': 2,          # 保持小批次以提高稳定性
        'num_epochs': 150,        # 增加训练轮数以获得更好的收敛
        'learning_rate': 0.00005, # 进一步降低学习率以适应两阶段训练
        'weight_decay': 1e-7,     # 进一步减少权重衰减
        'num_workers': 2,
        'train_split_ratio': 0.8
    }
    
    # 学习率调度器配置
    SCHEDULER_PARAMS = {
        'factor': 0.2,            # 增强学习率衰减
        'patience': 5,            # 减少耐心值
        'min_lr': 1e-10           # 更小的最小学习率
    }
    
    # 损失函数权重配置
    LOSS_WEIGHTS = {
        'diversity_weight': 0.3,  # 适度增加多样性权重以防止模型坍缩
        'contrastive_weight': 4.0, # 增加对比权重以强化正常和异常样本的区分
        'margin': 8.0              # 增加边界值以增强分离效果
    }
    
    # 检测器配置
    DETECTOR_PARAMS = {
        'threshold_percentile': 85  # 降低阈值百分位数以提高召回率
    }
    
    # 训练器配置
    TRAINER_PARAMS = {
        'early_stopping_patience': 30,
        'scheduler_factor': 0.2,
        'scheduler_patience': 5,
        'scheduler_min_lr': 1e-10,
        'gradient_clip_norm': 0.3   # 进一步降低梯度裁剪范数以适应更强的训练
    }
    
    # 可视化配置
    VISUALIZATION_PARAMS = {
        'num_examples': 3
    }