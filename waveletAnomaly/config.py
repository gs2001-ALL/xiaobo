# config.py
import torch

class Config:
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("DEVICE:",DEVICE)

    # 随机种子
    SEED = 15
    
    # 数据集路径配置
    # TRAIN_ROOT_DIR = '../../dataset/UCSD/UCSDped2/Train'
    # TEST_ROOT_DIR = '../../dataset/UCSD/UCSDped2/Test'
    # LABELS_DIR = '../../dataset/UCSD/UCSDped2/label'
    TRAIN_ROOT_DIR = 'E:/dataset/UCSD/UCSDped2/Train'
    TEST_ROOT_DIR = 'E:/dataset/UCSD/UCSDped2/Test'
    LABELS_DIR = 'E:/dataset/UCSD/UCSDped2/label'


    # 数据集分段配置
    DATASET_PARAMS = {
        'segment_length': 16,      # 每个视频段的帧数
        'overlap_ratio': 0.5       # 视频段重叠比例
    }

    # 数据预处理配置
    IMAGE_SIZE = (256, 256)  # 降低图像尺寸以减少计算量
    FILE_TYPE = '.tif'
    MAX_FRAMES = 50  # 减少最大帧数以加快训练
    
    # 模型配置
    MODEL_PARAMS = {
        'in_channels': 1,        # 输入通道数
        'hidden_channels': 32,   # 增加隐藏通道数以处理更复杂的特征
        'out_channels': 16,      # 修改：增加输出通道数以提高表达能力
        'grid_size': 8,          # 网格大小
        'temporal_window': 2,    # 时间窗口大小,节点前后两个点
    }
    
    # 小波变换配置
    WAVELET_PARAMS = {
        'wavelet': 'haar',
        'levels': 1  # 增加小波分解层数以获得更多细节信息
    }
    
    # 训练配置
    TRAINING_PARAMS = {
        'batch_size': 2,         # 修改：增加批次大小以提高训练稳定性
        'num_epochs': 150,       # 修改：增加训练轮数以获得更好的收敛
        'learning_rate': 0.01, # 显著降低学习率
        'weight_decay': 1e-7,    # 减少权重衰减
        'num_workers': 2,
        'train_split_ratio': 0.8
    }
    
    # 学习率调度器配置
    SCHEDULER_PARAMS = {
        'factor': 0.5,           # 修改：增加学习率衰减因子
        'patience': 10,          # 增加耐心值
        'min_lr': 1e-8          # 最小学习率
    }
    
    # 损失函数权重配置
    LOSS_WEIGHTS = {
        'diversity_weight': 0.1,  # 修改：适度增加多样性权重以防止模型坍缩
        'contrastive_weight': 1.0, # 修改：增加对比权重以更好地区分正常和异常样本
        'bce_weight': 0.5,         # 默认'
        'margin': 1.0,             # 修改：增加边界值以增强区分度,建议 0.5~2.0
        'infonce_weight': 2.0,   # 默认'
      
    }
    
    LOSS_PARAMS ={
        # Focal Loss parameters
        'focal_loss_alpha': 0.75,  # Weighting factor for class imbalance
        'focal_loss_gamma': 2.0   # Focusing para meter
    }

    # 检测器配置
    DETECTOR_PARAMS = {
        'threshold_percentile': 85  # 修改：增加阈值百分位数以提高精确率
    }
    
    # 训练器配置
    TRAINER_PARAMS = {
        'early_stopping_patience': 10,
        'scheduler_factor': 0.5,     # 修改：与SCHEDULER_PARAMS.factor保持一致
        'scheduler_patience': 10,
        'scheduler_min_lr': 1e-12,
        'gradient_clip_norm': 0.5    # 修改：增加梯度裁剪范数以提高训练稳定性
    }
    
    # 可视化配置
    VISUALIZATION_PARAMS = {
        'num_examples': 3
    }

     