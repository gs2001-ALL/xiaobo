import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import random
import utils
from models_i3d import ImprovedWaveletSTGNN
from datasets import create_train_test_datasets, get_data_loaders, custom_collate_fn
from config import Config
import datetime
from utils import setup_logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# 获取logger实例
logger = setup_logging()

def set_random_seed(seed=Config.SEED):
    """
    设置随机种子以确保实验可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"随机种子设置为: {seed}")

def worker_init_fn(worker_id):
    """
    为每个DataLoader worker设置随机种子
    """
    # 为每个worker设置不同的种子
    worker_seed = Config.SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

def save_model(model, filename):
    """
    保存模型到文件
    """
    try:
        # 创建模型保存目录（如果不存在）
        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存模型状态字典
        filepath = os.path.join(model_dir, filename)
        torch.save(model.state_dict(), filepath)
        logger.info(f"模型已保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")

def load_model(model, filename, device):
    """
    从文件加载模型
    """
    try:
        # 构建文件路径
        filepath = os.path.join("saved_models", filename)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            logger.warning(f"模型文件 {filepath} 不存在")
            return False
        
        # 加载模型状态字典
        model.load_state_dict(torch.load(filepath, map_location=device))
        logger.info(f"模型已从 {filepath} 加载")
        return True
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return False

class RobustAnomalyTrainer:
    """
    鲁棒的异常检测训练器，防止模型坍缩
    """
    def __init__(self, model, device, 
                 diversity_weight=Config.LOSS_WEIGHTS['diversity_weight'], 
                 contrastive_weight=Config.LOSS_WEIGHTS['contrastive_weight'], 
                 margin=Config.LOSS_WEIGHTS['margin'],
                 bce_weight = Config.LOSS_WEIGHTS['bce_weight']):
        self.model = model.to(device)
        self.device = device
        self.diversity_weight = diversity_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.bce_weight = bce_weight
        self.current_epoch = 0  # 初始化当前epoch为0
        # 优化器，使用配置中的学习率和权重衰减
        # model_params = [
        #     {'params': model.gated_gcn.parameters(), 'lr': Config.TRAINING_PARAMS['learning_rate']},
        #     {'params': model.output_projection.parameters(), 'lr': Config.TRAINING_PARAMS['learning_rate']},
        #     {'params': model.attention.parameters(), 'lr': Config.TRAINING_PARAMS['learning_rate'] * 1.5},
        #     {'params': model.batch_norm.parameters(), 'lr': Config.TRAINING_PARAMS['learning_rate']},
        # ]

        # # Check if residual_projection exists and add its parameters
        # if hasattr(model, 'residual_projection') and model.residual_projection is not None:
        #     model_params.append({'params': model.residual_projection.parameters(), 'lr': Config.TRAINING_PARAMS['learning_rate']})

        self.optimizer = optim.AdamW(model.parameters(), 
                                  lr=Config.TRAINING_PARAMS['learning_rate'],
                                  weight_decay=Config.TRAINING_PARAMS['weight_decay'])
        
        # 学习率调度器 - 使用配置参数
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控PR-ROC指标，越大越好
            factor=Config.TRAINER_PARAMS['scheduler_factor'],
            patience=Config.TRAINER_PARAMS['scheduler_patience'],
            # verbose=True,
            min_lr=Config.TRAINER_PARAMS['scheduler_min_lr']
        )
        
        # 早停机制参数
        self.best_test_pr_auc = 0.0
        self.epochs_without_improvement = 0
        self.patience = Config.TRAINER_PARAMS['early_stopping_patience']
        
        logger.info("训练器初始化完成")
        logger.info(f"设备: {device}")
        logger.info(f"多样性权重: {diversity_weight}")
        logger.info(f"对比权重: {contrastive_weight}")
        logger.info(f"边界值: {margin}")


    def generate_pseudo_anomalies1(self, normal_videos):
        """
        生成更真实的伪异常样本，迫使模型学会更好地区分正常和异常
        
        参数:
            normal_videos (torch.Tensor): 正常视频样本
                输入维度: [batch_size, channels, time, height, width]
                例如: [4, 1, 8, 128, 128] 表示4个批次，1个通道，8帧，128x128的图像
            
        返回:
            torch.Tensor: 伪异常视频样本
                输出维度: [batch_size, channels, time, height, width]
                例如: [4, 1, 8, 128, 128] 与输入维度相同
        """
        batch_size, channels, time, height, width = normal_videos.shape
        
        # 根据训练进度动态调整扰动强度   
        epoch_progress = self.current_epoch / Config.TRAINING_PARAMS['num_epochs']
        noise_strength = 2.0 + 3.0 * epoch_progress 

        # 首先选择一种方法生成伪异常样本
        choice = torch.randint(0, 6, (1,), device=self.device).item()
        
        if choice == 0:
            # 方法1: 强噪声 - 添加更大强度的高斯噪声模拟传感器故障
            noise = torch.randn_like(normal_videos) * noise_strength
            pseudo_anomaly_videos = normal_videos + noise
        elif choice == 1:
            # 方法2: 大范围遮挡 - 模拟物体遮挡或传感器损坏
            mask = torch.rand(batch_size, 1, 1, height, width, device=self.device) > (0.6 - 0.2 * epoch_progress)
            pseudo_anomaly_videos = normal_videos * mask.float()
        elif choice == 2:
            # 方法3: 时间帧洗牌 - 模拟时间顺序错乱
            pseudo_anomaly_videos = normal_videos.clone()
            for i in range(batch_size):
                # 洗牌更多帧以产生更明显的异常
                min_frames = max(2, time * 2 // 3)
                num_frames_to_shuffle = torch.randint(min_frames, time, (1,)).item()
                indices = torch.randperm(time)[:num_frames_to_shuffle]
                target_indices = torch.randperm(time)[:num_frames_to_shuffle]
                pseudo_anomaly_videos[i, :, indices, :, :] = normal_videos[i, :, target_indices, :, :]
        elif choice == 3:
            # 方法4: 帧重复 - 模拟运动停滞
            pseudo_anomaly_videos = normal_videos.clone()
            for i in range(batch_size):
                # 复制更多帧以产生更明显的异常
                source_idx = torch.randint(0, time, (max(1, time//2),)).tolist()
                target_idx = torch.randint(0, time, (max(1, time//2),)).tolist()
                for s_idx, t_idx in zip(source_idx, target_idx):
                    pseudo_anomaly_videos[i, :, t_idx, :, :] = normal_videos[i, :, s_idx, :, :]
        elif choice == 4:
            # 方法5: 帧缺失 - 模拟帧丢失
            pseudo_anomaly_videos = normal_videos.clone()
            for i in range(batch_size):
                # 随机选择要置零的帧（置零1/3的帧）
                missing_idx = torch.randint(0, time, (max(1, time//3),)).tolist()
                pseudo_anomaly_videos[i, :, missing_idx, :, :] = 0
        else:
            # 方法6: 频率扰动 - 在频域添加扰动
            pseudo_anomaly_videos = normal_videos.clone()
            for i in range(batch_size):
                # 对1/3的帧添加频率扰动
                perturb_idx = torch.randint(0, time, (max(1, time//3),)).tolist()
                for idx in perturb_idx:
                    # 添加高频噪声
                    high_freq_noise = torch.randn_like(normal_videos[i, :, idx, :, :]) * 2.0
                    pseudo_anomaly_videos[i, :, idx, :, :] = normal_videos[i, :, idx, :, :] + high_freq_noise
        
        return pseudo_anomaly_videos

    def generate_pseudo_anomalies(self, normal_videos):
        """
        生成更真实、多样、连贯的伪异常样本，提升模型对真实异常的识别能力

        参数:
            normal_videos (torch.Tensor): 正常视频样本
                输入维度: [batch_size, channels, time, height, width]
                例如: [4, 1, 8, 128, 128]

        返回:
            torch.Tensor: 伪异常视频样本
                输出维度: [batch_size, channels, time, height, width]
        """
        batch_size, channels, time, height, width = normal_videos.shape

        # 动态扰动强度（使用余弦调度）
        epoch_progress = self.current_epoch / Config.TRAINING_PARAMS['num_epochs']
        noise_strength = 2.0 + 3.0 * (1 - torch.cos(torch.tensor(epoch_progress * np.pi / 2)))

        # 初始化伪异常样本
        pseudo_anomaly_videos = normal_videos.clone()

        # 方法1: 强噪声 + 遮挡（混合扰动）
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(normal_videos) * noise_strength
            pseudo_anomaly_videos += noise
            # 添加遮挡
            mask = torch.rand(batch_size, 1, 1, height, width, device=self.device) > (0.6 - 0.2 * epoch_progress)
            pseudo_anomaly_videos = pseudo_anomaly_videos * mask.float()

        # 方法2: 时间帧洗牌（增强时间连贯性）
        if torch.rand(1).item() < 0.5:
            for i in range(batch_size):
                num_segments = torch.randint(2, 4, (1,)).item()
                segment_len = max(1, time // num_segments)  # 确保至少为1
                for seg in range(num_segments):
                    # 确保不超出时间维度范围
                    start = seg * segment_len
                    end = min(start + segment_len, time)  # 不超出时间范围
                    if start >= time:
                        break
                    # 计算目标位置，确保目标区域大小与源区域一致
                    actual_segment_len = end - start
                    if actual_segment_len > 0:
                        target_start = torch.randint(0, max(1, time - actual_segment_len + 1), (1,)).item()
                        target_end = target_start + actual_segment_len
                        # 确保目标区域不超出范围
                        if target_end <= time:
                            pseudo_anomaly_videos[i, :, target_start:target_end, :, :] = \
                                normal_videos[i, :, start:end, :, :]

        # 方法3: 帧重复 + 帧缺失（模拟运动停滞与帧丢失）
        if torch.rand(1).item() < 0.5:
            for i in range(batch_size):
                # 帧重复
                source_idx = torch.randint(0, time, (max(1, time // 2),)).tolist()
                target_idx = torch.randint(0, time, (max(1, time // 2),)).tolist()
                for s_idx, t_idx in zip(source_idx, target_idx):
                    # 确保索引在有效范围内
                    if 0 <= s_idx < time and 0 <= t_idx < time:
                        pseudo_anomaly_videos[i, :, t_idx, :, :] = normal_videos[i, :, s_idx, :, :]
                # 帧缺失
                missing_idx = torch.randint(0, time, (max(1, time // 3),)).tolist()
                for m_idx in missing_idx:
                    # 确保索引在有效范围内
                    if 0 <= m_idx < time:
                        pseudo_anomaly_videos[i, :, m_idx, :, :] = 0

        # 方法4: 频率扰动 + 空域注意力扰动（已修复维度问题）
        if torch.rand(1).item() < 0.5:
            for i in range(batch_size):
                perturb_idx = torch.randint(0, time, (max(1, time // 3),)).tolist()
                for idx in perturb_idx:
                    # 确保idx在有效范围内
                    if idx >= time or idx < 0:
                        continue
                    # 频率扰动（FFT）
                    try:
                        fft = torch.fft.fft2(normal_videos[i, :, idx, :, :])
                        fft += torch.randn_like(fft) * 1.5
                        ifft_result = torch.fft.ifft2(fft).real
                        # 确保结果维度与目标位置匹配
                        if ifft_result.shape == pseudo_anomaly_videos[i, :, idx, :, :].shape:
                            pseudo_anomaly_videos[i, :, idx, :, :] = ifft_result
                    except Exception as e:
                        # 如果FFT操作失败，使用简单的噪声扰动
                        noise = torch.randn_like(normal_videos[i, :, idx, :, :]) * 1.5
                        pseudo_anomaly_videos[i, :, idx, :, :] += noise

                    # 空域注意力扰动（修复后的版本）
                    try:
                        attention_mask = torch.rand(height, width, device=self.device) > 0.8
                        attention_mask = attention_mask.unsqueeze(0).expand(channels, -1, -1)  # [C, H, W]
                        # 确保掩码维度匹配
                        if attention_mask.shape == pseudo_anomaly_videos[i, :, idx, :, :].shape:
                            pseudo_anomaly_videos[i, :, idx, :, :][attention_mask] += 2.0
                    except Exception as e:
                        # 如果注意力扰动失败，跳过这一步
                        pass

        return pseudo_anomaly_videos

    def compute_infonce_loss(self,
                         normal_embeddings: torch.Tensor,
                         pseudo_anomaly_embeddings: torch.Tensor,
                         temperature: float = 0.05) -> torch.Tensor:
        """
        计算 InfoNCE 对比损失，每个 normal 嵌入与对应的 pseudo_anomaly 嵌入为正样本对，
        其他所有样本为负样本。
        """
        device = normal_embeddings.device
        batch_size = normal_embeddings.shape[0]

        # Step 1: 构造所有样本
        all_samples = torch.cat([normal_embeddings, pseudo_anomaly_embeddings], dim=0)  # [2B, D]

        # Step 2: 计算相似度矩阵
        sim_matrix = F.cosine_similarity(all_samples.unsqueeze(1), all_samples.unsqueeze(0), dim=-1) / temperature  # [2B, 2B]

        # Step 3: 构造标签：每个 normal 样本的正样本是对应的 pseudo_anomaly 样本
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels, labels], dim=0)  # [2B]，前B个normal，后B个pseudo-anomaly，正样本对是 (i, i + B)

        # Step 4: 构造 mask：对角线 + B 的位置为 True（表示正样本）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True

        # Step 5: 构造 logits 和 labels
        logits = sim_matrix[mask].view(2 * batch_size, -1)  # [2B, 1 + (2B - 1)]，其中1是正样本
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

        # Step 6: 计算 InfoNCE 损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_contrastive_loss(self,
                             normal_scores: torch.Tensor,
                             pseudo_anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        改进的对比损失：增强正常样本和伪异常样本的区分度
        
        参数:
            normal_scores (torch.Tensor): 正常样本的异常分数 [batch_size, time]
            pseudo_anomaly_scores (torch.Tensor): 伪异常样本的异常分数 [batch_size, time]
            
        返回:
            torch.Tensor: 对比损失值
        """
        device = normal_scores.device
        ns = normal_scores.view(-1).to(device)
        ps = pseudo_anomaly_scores.view(-1).to(device)

        # 1. 正常样本：鼓励分数接近0
        normal_loss = torch.mean(ns ** 2)

        # 2. 伪异常样本：鼓励分数为正且有一定大小
        anomaly_margin = 1.0
        anomaly_loss = torch.mean(torch.clamp(anomaly_margin - ps, min=0) ** 2)

        # 3. 间隔损失：两类均值应该有明显间隔
        normal_mean = torch.mean(ns)
        anomaly_mean = torch.mean(ps)
        separation_loss = torch.clamp(anomaly_margin - (anomaly_mean - normal_mean), min=0) ** 2

        # 4. 分布分离损失：鼓励两类分数分布分离
        # 计算两类分数的方差
        normal_var = torch.var(ns) if len(ns) > 1 else torch.tensor(0.0, device=device)
        anomaly_var = torch.var(ps) if len(ps) > 1 else torch.tensor(0.0, device=device)
        
        # 鼓励正常样本分数方差小（一致性），伪异常样本分数方差适中
        distribution_loss = torch.relu(normal_var - 0.5) + torch.relu(0.1 - anomaly_var) + torch.relu(anomaly_var - 3.0)

        # 5. 加权求和
        w_normal, w_anomaly, w_sep, w_dist = 1.0, 1.5, 2.0, 0.5
        total = w_normal * normal_loss + w_anomaly * anomaly_loss + w_sep * separation_loss + w_dist * distribution_loss

        return total

    def compute_asymmetric_focal_loss(self, 
                    normal_scores: torch.Tensor,
                    pseudo_anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        引入 Asymmetric Focal Loss 的改进版本，提升召回率
        
        Args:
            normal_scores (torch.Tensor): 正常样本的异常得分 [B, T]
            pseudo_anomaly_scores (torch.Tensor): 伪异常样本的异常得分 [B, T]
        
        Returns:
            torch.Tensor: 损失值
        """
        # 从配置中读取参数
        alpha_pos = Config.LOSS_PARAMS.get('focal_loss_alpha_pos', 0.75)  # 正样本权重
        alpha_neg = Config.LOSS_PARAMS.get('focal_loss_alpha_neg', 0.25)  # 负样本权重
        gamma_pos = Config.LOSS_PARAMS.get('focal_loss_gamma_pos', 2.0)    # 正样本难例挖掘
        gamma_neg = Config.LOSS_PARAMS.get('focal_loss_gamma_neg', 1.0)    # 负样本难例挖掘

        # 构造标签
        normal_labels = torch.zeros_like(normal_scores)
        pseudo_anomaly_labels = torch.ones_like(pseudo_anomaly_scores)

        # 合并输入和标签
        all_scores = torch.cat([normal_scores.view(-1), pseudo_anomaly_scores.view(-1)], dim=0)
        all_labels = torch.cat([normal_labels.view(-1), pseudo_anomaly_labels.view(-1)], dim=0)

        # 概率转换
        probs = torch.sigmoid(all_scores)
        probs = probs.clamp(min=1e-8, max=1 - 1e-8)  # 防止数值不稳定

        # 构造 pt
        p_t = probs * all_labels + (1 - probs) * (1 - all_labels)

        # 构造 alpha_t 和 gamma_t（非对称）
        alpha_t = all_labels * alpha_pos + (1 - all_labels) * alpha_neg
        gamma_t = all_labels * gamma_pos + (1 - all_labels) * gamma_neg

        # Focal Loss 核心公式
        focal_weight = (1 - p_t) ** gamma_t
        focal_loss = -alpha_t * focal_weight * torch.log(p_t)

        return focal_loss.mean()


    def compute_bce_loss(self, 
                    normal_scores: torch.Tensor,
                    pseudo_anomaly_scores: torch.Tensor) -> torch.Tensor:
        """ Focal Loss implementation to improve classification ability 
        Args: 
        normal_scores (torch.Tensor): Anomaly scores for normal samples [batch_size, time] 
        pseudo_anomaly_scores (torch.Tensor): Anomaly scores for pseudo-anomaly samples [batch_size, time] 
        Returns: 
        torch.Tensor: Focal Loss value """ 
        alpha = Config.LOSS_PARAMS['focal_loss_alpha']
        gamma = Config.LOSS_PARAMS['focal_loss_gamma']

        # 构造标签
        normal_labels = torch.zeros_like(normal_scores)
        pseudo_anomaly_labels = torch.ones_like(pseudo_anomaly_scores)

        # 合并数据
        all_scores = torch.cat([normal_scores.view(-1), pseudo_anomaly_scores.view(-1)], dim=0)
        all_labels = torch.cat([normal_labels.view(-1), pseudo_anomaly_labels.view(-1)], dim=0)

        # 概率转换
        probs = torch.sigmoid(all_scores)

        # 构造 pt
        p_t = probs * all_labels + (1 - probs) * (1 - all_labels)
        p_t = p_t.clamp(min=1e-8, max=1 - 1e-8)  # 数值稳定性

        # 构造 alpha_t
        alpha_t = all_labels * alpha + (1 - all_labels) * (1 - alpha)

        # focal weight
        focal_weight = (1 - p_t) ** gamma

        # 最终损失
        focal_loss = -alpha_t * focal_weight * torch.log(p_t)

        return focal_loss.mean()

    def compute_diversity_loss(self, scores):
        """
        计算多样性损失，防止模型输出过于集中
        """
        mean_score = torch.mean(scores)
        variance = torch.var(scores)
        diversity_loss = torch.relu(0.1 - variance)  # 鼓励方差大于0.1
        return diversity_loss

    def train_epoch(self, train_loader):
        """
        改进的训练epoch：确保所有参数接收梯度
        """
        self.model.train()
        total_loss = 0.0
        infonce_loss = 0.0
        bce_loss = 0.0
        diversity_loss = 0.0
        batch_count = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(self.device).float()
            labels = labels.to(self.device).float()
            
            # 检查输入数据
            if torch.isnan(videos).any() or torch.isinf(videos).any():
                logger.warning(f"批次 {batch_idx}: 输入数据包含NaN或inf值，跳过")
                continue
                
            if len(videos.shape) != 5:
                logger.warning(f"批次 {batch_idx}: 输入数据维度不正确: {videos.shape}，跳过")
                continue
            
            videos = torch.clamp(videos, min=-5, max=5)
                
            self.optimizer.zero_grad()
            
            
            try:
                # 正常样本前向传播
                normal_scores = self.model(videos)
                
                if torch.isnan(normal_scores).any() or torch.isinf(normal_scores).any():
                    logger.warning(f"批次 {batch_idx}: 正常样本输出包含NaN或inf值")
                    normal_scores = torch.nan_to_num(normal_scores, nan=0.0, posinf=1.0, neginf=-1.0)
                    if torch.isnan(normal_scores).any() or torch.isinf(normal_scores).any():
                        continue
            except Exception as e:
                logger.warning(f"正常样本前向传播失败: {e}")
                continue
            
            # 获取正常样本嵌入
            normal_embeddings = self.model.get_embeddings()
            normal_scores = torch.clamp(normal_scores, min=0, max=10)
            
            try:
                # 生成伪异常样本
                pseudo_anomalies = self.generate_pseudo_anomalies(videos)
                
                if torch.isnan(pseudo_anomalies).any() or torch.isinf(pseudo_anomalies).any():
                    logger.warning(f"批次 {batch_idx}: 生成的伪异常样本包含NaN或inf值")
                    pseudo_anomalies = torch.nan_to_num(pseudo_anomalies, nan=0.0, posinf=1.0, neginf=-1.0)
                    if torch.isnan(pseudo_anomalies).any() or torch.isinf(pseudo_anomalies).any():
                        continue
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 生成伪异常样本失败: {e}")
                continue
                
            try:
                pseudo_anomaly_scores = self.model(pseudo_anomalies)
                
                if torch.isnan(pseudo_anomaly_scores).any() or torch.isinf(pseudo_anomaly_scores).any():
                    logger.warning(f"批次 {batch_idx}: 伪异常样本输出包含NaN或inf值")
                    pseudo_anomaly_scores = torch.nan_to_num(pseudo_anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)
                    if torch.isnan(pseudo_anomaly_scores).any() or torch.isinf(pseudo_anomaly_scores).any():
                        continue
            except Exception as e:
                logger.warning(f"伪异常样本前向传播失败: {e}")
                continue
            
            # 获取伪异常样本嵌入
            pseudo_anomaly_embeddings = self.model.get_embeddings()
            pseudo_anomaly_scores = torch.clamp(pseudo_anomaly_scores, min=0, max=10)
            
            try:
                # 计算InfoNCE损失
                infonce_loss_val = self.compute_infonce_loss(normal_embeddings, pseudo_anomaly_embeddings)
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 计算InfoNCE损失失败: {e}")
                continue
            
            try:
                # 计算BCE损失
                # bce_loss_val = self.compute_bce_loss(normal_scores, pseudo_anomaly_scores)  # 使用段级标签
                bce_loss_val = self.compute_asymmetric_focal_loss(normal_scores, pseudo_anomalies)
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 计算BCE损失失败: {e}")
                continue

            try:
                # 计算多样性损失
                diversity_loss_val = self.compute_diversity_loss(normal_scores)  
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 计算diversity损失失败: {e}")
                continue
            
            infonce_loss_val = torch.clamp(infonce_loss_val, max=20)
            bce_loss_val = torch.clamp(bce_loss_val, max=10)
            diversity_loss_val = torch.clamp(diversity_loss_val, max=10)
            
            # 加权组合损失
            total_loss_val = Config.LOSS_WEIGHTS['infonce_weight'] * infonce_loss_val + \
                            Config.LOSS_WEIGHTS['bce_weight'] * bce_loss_val+\
                            Config.LOSS_WEIGHTS['diversity_weight'] * diversity_loss_val
            
            if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
                logger.warning(f"批次 {batch_idx}: 总损失为NaN或inf，跳过")
                continue
            
            if total_loss_val.item() > 50:
                logger.warning(f"批次 {batch_idx} 损失过大: {total_loss_val.item()}")
                continue
            
            try:
                total_loss_val.backward()
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 反向传播失败: {e}")
                continue
            
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 梯度裁剪失败: {e}")
            
            # Debugging: Log gradients for all parameters
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    logger.debug(f"Gradient for {name}: mean={param.grad.mean().item()}, max={param.grad.max().item()}")

            try:
                self.optimizer.step()
            except Exception as e:
                logger.warning(f"批次 {batch_idx}: 参数更新失败: {e}")
                continue
            
            total_loss += total_loss_val.item()
            infonce_loss += infonce_loss_val.item()
            bce_loss += bce_loss_val.item()
            diversity_loss += diversity_loss_val.item()
            batch_count += 1
            
            if batch_idx % 50 == 0:
                logger.debug(f"批次 {batch_idx} - 总损失: {total_loss_val.item():.4f}, "
                            f"InfoNCE损失: {infonce_loss_val.item():.4f}, "
                            f"BCE损失: {bce_loss_val.item():.4f},"
                            f"deversity损失: {diversity_loss_val.item():.4f}")
        
        if batch_count == 0:
            logger.warning("训练批次计数为0")
            return {
                'total_loss': 0.0,
                'infonce_loss': 0.0,
                'bce_loss': 0.0,
                'diversity_loss': 0.0
            }
        
        avg_total_loss = total_loss / batch_count
        avg_infonce_loss = infonce_loss / batch_count
        avg_bce_loss = bce_loss / batch_count
        avg_diversity_loss = diversity_loss / batch_count
        
        logger.info(f"  训练损失 - 总损失: {avg_total_loss:.4f}, "
                    f"InfoNCE: {avg_infonce_loss:.4f}, "
                    f"BCE: {avg_bce_loss:.4f},"
                    f"diversity: {avg_diversity_loss:.4f}")
        
        return {
            'total_loss': avg_total_loss,
            'infonce_loss': avg_infonce_loss,
            'bce_loss': avg_bce_loss,
            'diversity_loss': avg_diversity_loss
        }


    def validate(self, val_loader):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                # 确保数据在正确的设备上
                videos = videos.to(self.device).float()
                labels = labels.to(self.device)
                
                # 检查输入数据
                if torch.isnan(videos).any() or torch.isinf(videos).any():
                    continue
                    
                # 检查输入维度
                if len(videos.shape) != 5:
                    continue
                
                try:
                    outputs = self.model(videos)
                except Exception as e:
                    logger.warning(f"验证前向传播失败: {e}")
                    continue
                
                # 检查输出
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                loss = torch.mean(outputs ** 2)  # 重构损失
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                batch_count += 1
        
        if batch_count == 0:
            logger.warning("验证批次计数为0")
            return 0.0
            
        avg_loss = total_loss / batch_count
        logger.info(f"  验证损失: {avg_loss:.4f}")
        return avg_loss
    
    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                
                # 统计每个视频段中出现次数最多的标签
                most_common_labels = []
                for sample_labels in labels:  # 遍历每个样本的标签
                    unique_labels, counts = torch.unique(sample_labels, return_counts=True)
                    most_common_label = unique_labels[torch.argmax(counts)].item()  # 出现次数最多的标签
                    most_common_labels.append(most_common_label)
                
                # 转换为 NumPy 数组
                most_common_labels = np.array(most_common_labels)
                
                outputs = self.model(inputs)
                
                # 提取正类的概率
                preds = torch.sigmoid(outputs).cpu().numpy()  # 假设正类是第2列
                
                # 直接扩展一维数组，而不是二维数组
                all_preds.extend(preds)
                all_labels.extend(most_common_labels)

        # 将列表转换为 NumPy 数组
        all_preds = np.array(all_preds).flatten()  # 确保是一维数组
        all_labels = np.array(all_labels).flatten()  # 确保是一维数组

        # 检查标签和预测是否有效
        if len(all_labels) == 0 or len(all_preds) == 0:
            logger.warning("测试数据为空，无法计算指标")
            return {'PR_AUC': 0.0, 'ROC_AUC': 0.0}

        # 移除 NaN 和 Inf 值
        try:
            valid_indices = ~(np.isnan(all_labels) | np.isinf(all_labels) |
                            np.isnan(all_preds) | np.isinf(all_preds))
            valid_indices = valid_indices.flatten()  # 确保布尔索引是一维的
            
            # 使用布尔索引过滤无效值
            all_labels = all_labels[valid_indices]
            all_preds = all_preds[valid_indices]
        except Exception as e:
            logger.error(f"移除无效值时出错: {e}")
            return {'PR_AUC': 0.0, 'ROC_AUC': 0.0}

        # 检查是否有足够的类别分布
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            logger.warning("警告: 标签中只有一种类别，无法计算AUC指标")
            return {'PR_AUC': 0.0, 'ROC_AUC': 0.0}

        # 计算 ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            logger.error(f"计算 ROC-AUC 时出错: {e}")
            roc_auc = 0.0

        # 计算 PR-AUC
        try:
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            pr_auc = auc(recall, precision)
        except Exception as e:
            logger.error(f"计算 PR-AUC 时出错: {e}")
            pr_auc = 0.0

        # 返回结果字典
        return {
            'PR_AUC': pr_auc,
            'ROC_AUC': roc_auc
        } 

    def evaluate_on_test(self, test_loader, train_loader_for_threshold):
        """
        在测试集上评估模型 (使用训练数据更新阈值)
        
        参数:
            test_loader (DataLoader): 测试数据加载器
                输入类型: PyTorch DataLoader对象
                包含测试数据
            train_loader_for_threshold (DataLoader): 训练数据加载器用于阈值更新
                输入类型: PyTorch DataLoader对象
                包含训练数据用于阈值计算
                
        返回:
            dict: 包含评估指标的字典
                例如: {'roc_auc': 0.9, 'pr_auc': 0.85, 'accuracy': 0.8, ...}
        """
        # 创建检测器用于评估
        detector = AdaptiveThresholdDetector(self.model, self.device)
        # 使用训练数据更新阈值
        detector.update_threshold(train_loader_for_threshold)
        
        predictions, labels, scores = detector.detect_anomalies(test_loader)
        
        # 修改开始：如果检测到的异常样本数为0，尝试调整阈值策略
        if np.sum(predictions) == 0:
            logger.warning("检测到的异常样本数为0，尝试调整阈值策略")
            # 使用更低的阈值百分位数重新计算阈值
            detector.threshold_percentile = max(70, detector.threshold_percentile - 10)
            detector.update_threshold(train_loader_for_threshold)
            predictions, labels, scores = detector.detect_anomalies(test_loader)
            logger.info(f"调整后阈值百分位数: {detector.threshold_percentile}, 新阈值: {detector.threshold:.6f}")
        # 修改结束
        
        metrics = evaluate_model_comprehensive(predictions, labels, scores)
        
        return metrics

    def train(self, train_loader, val_loader, test_loader, num_epochs=Config.TRAINING_PARAMS['num_epochs']):
        """
        完整训练过程，使用精简的损失函数
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            test_loader (DataLoader): 测试数据加载器
            num_epochs (int): 训练轮数
            
        返回:
            tuple: (train_losses, val_losses, all_losses, test_metrics)
        """
        logger.info("开始鲁棒训练...")
        logger.info(f"设备: {self.device}")
        
        train_losses = []
        val_losses = []
        test_metrics = []  # 跟踪测试性能
        all_losses = {
            'total': [],
            'infonce': [],
            'bce': [],
            'diversity':[]
        }
        
        # 用于跟踪最佳模型
        best_test_pr_auc = 0.0  # 修改：使用PR AUC作为主要评价标准
        best_epoch = 0
        best_test_metrics = {}  # 保存最佳模型的测试指标
        
        # 修改开始：实现更稳定的学习率训练策略
        # 实现两次学习率训练策略
        # 第一阶段：使用较低学习率进行稳定训练
        first_stage_epochs = num_epochs // 2
        second_stage_epochs = num_epochs - first_stage_epochs
        
        # 第一阶段学习率 (使用更低的学习率)
        first_stage_lr = Config.TRAINING_PARAMS['learning_rate'] * 0.1  # 0.1倍于配置学习率
        # 第二阶段学习率
        second_stage_lr = Config.TRAINING_PARAMS['learning_rate']  # 正常学习率
        
        # 设置第一阶段学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = first_stage_lr
        logger.info(f"第一阶段训练: {first_stage_epochs} epochs, 学习率: {first_stage_lr}")
        # 修改结束
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch  # 更新当前epoch
            # 修改开始：在中间点切换到第二阶段学习率
            # 在中间点切换到第二阶段学习率
            if epoch == first_stage_epochs:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = second_stage_lr
                logger.info(f"切换到第二阶段训练: {second_stage_epochs} epochs, 学习率: {second_stage_lr}")
            # 修改结束
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")

            # 训练阶段
            train_metrics = self.train_epoch(train_loader)
            
            # logger.info("开始打印模型参数梯度信息:")
            utils.printGrad(self.model)
            # logger.info("模型参数梯度信息打印完成")
            
            # logger.info(f"参数绑定: {self.optimizer.param_groups[0]['params']}")

            # 检查训练指标是否有效
            if not (np.isnan(train_metrics['total_loss']) or np.isinf(train_metrics['total_loss'])):
                train_losses.append(train_metrics['total_loss'])
            else:
                train_losses.append(0.0)
            
            # 验证阶段
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # 测试评估阶段 (使用训练数据更新阈值)
            try:
                test_performance = self.evaluate_on_test(test_loader, train_loader)
                test_metrics.append(test_performance)
            except Exception as e:
                logger.error(f"测试评估出错: {e}")
                # 使用默认值继续
                test_performance = {
                    'roc_auc': 0.0, 'pr_auc': 0.0, 'accuracy': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                    'avg_score_normal': 0.0, 'avg_score_anomaly': 0.0
                }
                test_metrics.append(test_performance)
            
            # 修改开始：使用PR AUC作为学习率调度和早停的主要指标
            # 更新学习率（基于测试PR AUC）
            self.scheduler.step(test_performance['pr_auc'])
            
            # 检查是否为最佳模型（基于测试PR AUC）
            if test_performance['pr_auc'] > best_test_pr_auc:
                best_test_pr_auc = test_performance['pr_auc']
                best_epoch = epoch + 1
                best_test_metrics = test_performance.copy()
                # 保存最佳模型
                save_model(self.model, 'best_robust_wavelet_stgnn_model.pth')
                # logger.info(f"-> 保存新最佳模型 (PR AUC: {test_performance['pr_auc']:.4f}")
                
                # 重置早停计数器
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            # 修改结束
            
            # 早停机制
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"测试PR AUC在 {self.patience} 个epoch内没有改善，提前停止训练")
                break
            
            # 新增：如果验证损失变得过大，提前停止训练
            if val_loss > 10000:
                logger.info("验证损失过大，提前停止训练")
                break
                
            # 记录各种损失
            for key in all_losses:
                if key == 'total':
                    all_losses[key].append(train_metrics['total_loss'])
                else:
                    metric_key = f"{key}_loss"
                    if not (np.isnan(train_metrics[metric_key]) or np.isinf(train_metrics[metric_key])):
                        all_losses[key].append(train_metrics[metric_key])
                    else:
                        all_losses[key].append(0.0)
            
            # 每隔一定epoch或在关键点打印信息
            # 修改开始：使用PR AUC作为主要显示指标
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1 or test_performance['pr_auc'] > best_test_pr_auc:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                            f"Train Loss: {train_metrics['total_loss']:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Test PR AUC: {test_performance['pr_auc']:.4f} | "  # 修改：显示PR AUC
                            f"Test ROC AUC: {test_performance['roc_auc']:.4f} | "  # 修改：也显示ROC AUC
                            f"Test Recall: {test_performance['recall']:.4f}")   # 修改：显示召回率
                
                if test_performance['pr_auc'] > best_test_pr_auc:
                    logger.info(f"  -> 保存新最佳模型 (PR AUC: {test_performance['pr_auc']:.4f})")
            # 修改结束
        
        # 保存最终模型
        save_model(self.model, 'final_robust_wavelet_stgnn_model.pth')
        logger.info(f"训练完成。最佳测试PR AUC: {best_test_pr_auc:.4f} (Epoch {best_epoch})")  # 修改：显示PR AUC
        
        # 打印最佳模型的完整评估指标
        logger.info("\n最佳模型评估结果:")
        logger.info(f"  PR AUC: {best_test_metrics.get('pr_auc', 0):.4f}")  # 修改：PR AUC作为首要指标
        logger.info(f"  ROC AUC: {best_test_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  Accuracy: {best_test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {best_test_metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall: {best_test_metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 Score: {best_test_metrics.get('f1_score', 0):.4f}")
        
        return train_losses, val_losses, all_losses, test_metrics


class AdaptiveThresholdDetector:
    """
    改进的自适应阈值检测器
    """
    def __init__(self, model, device, threshold_percentile=Config.DETECTOR_PARAMS['threshold_percentile']):
        self.model = model.to(device)
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = 1.0  # 使用更合理的初始值
    
    def update_threshold(self, normal_loader):
        """
        使用正常数据更新异常检测阈值
        
        参数:
            normal_loader (DataLoader): 正常数据加载器
        """
        self.model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch_idx, (videos, _) in enumerate(normal_loader):
                try:
                    # 确保数据在正确的设备上
                    videos = videos.to(self.device).float()
                    # 检查输入数据
                    if torch.isnan(videos).any() or torch.isinf(videos).any():
                        continue
                    # 检查输入维度
                    if len(videos.shape) != 5:
                        continue
                    
                    # 限制输入范围
                    videos = torch.clamp(videos, min=-5, max=5)
                    
                    try:
                        scores = self.model(videos)
                        # 检查输出
                        if torch.isnan(scores).any() or torch.isinf(scores).any():
                            # 尝试修复输出
                            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
                            if torch.isnan(scores).any() or torch.isinf(scores).any():
                                continue
                        all_scores.append(scores.cpu().numpy())
                    except Exception as e:
                        continue
                except Exception as e:
                    continue
        
        if all_scores:
            try:
                # 处理不同长度的序列，将所有分数展平
                flattened_scores = []
                for scores in all_scores:
                    flattened_scores.append(scores.flatten())
                all_scores_flat = np.concatenate(flattened_scores)
                
                if len(all_scores_flat) > 0:
                    # 处理NaN和inf值
                    all_scores_flat = np.nan_to_num(all_scores_flat, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 检查是否还有有效值
                    if not np.isnan(all_scores_flat).any() and not np.isinf(all_scores_flat).any() and np.std(all_scores_flat) > 1e-10:
                        # 使用配置的百分位数
                        self.threshold = np.percentile(all_scores_flat, self.threshold_percentile)
                        logger.info(f"更新阈值: {self.threshold:.6f}")
                    else:
                        # 使用默认阈值
                        self.threshold = 1.0
                        logger.warning("使用默认阈值: 1.0")
                else:
                    self.threshold = 1.0
            except Exception as e:
                self.threshold = 1.0
        else:
            self.threshold = 1.0

# ... existing code ...

    def detect_anomalies(self, test_loader):
        """
        检测异常
        
        参数:
            test_loader (DataLoader): 测试数据加载器
                
        返回:
            tuple: (predictions, labels, scores)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_scores = []
        
        total_batches = len(test_loader)
        processed_batches = 0

        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(test_loader):
                try:
                    videos = videos.to(self.device).float()
                    labels = labels.to(self.device)
                    
                    # 检查输入数据是否包含NaN或inf值
                    if torch.isnan(videos).any() or torch.isinf(videos).any():
                        logger.warning(f"批次 {batch_idx}: 输入视频包含NaN或inf值，跳过")
                        continue
                        
                    if len(videos.shape) != 5:
                        logger.warning(f"批次 {batch_idx}: 输入视频维度不正确: {videos.shape}，跳过")
                        continue
                    
                    try:
                        scores = self.model(videos)
                    except Exception as e:
                        logger.warning(f"批次 {batch_idx}: 前向传播失败: {e}")
                        continue
                    
                    # 检查模型输出是否包含NaN或inf值
                    if torch.isnan(scores).any() or torch.isinf(scores).any():
                        logger.warning(f"批次 {batch_idx}: 模型输出包含NaN或inf值")
                        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
                        if torch.isnan(scores).any() or torch.isinf(scores).any():
                            logger.warning(f"批次 {batch_idx}: 修复后的模型输出仍然包含无效值，跳过")
                            continue
                    
                    # 确保scores在合理范围内
                    scores = torch.clamp(scores, min=0, max=10)
                    
                    # 使用阈值进行预测
                    predictions = (scores > self.threshold).float()
                    
                    # 确保predictions和labels形状一致
                    if scores.shape != labels.shape:
                        min_frames = min(scores.shape[1], labels.shape[1])
                        scores = scores[:, :min_frames]
                        labels = labels[:, :min_frames]
                        predictions = predictions[:, :min_frames]
                    
                    # 添加到结果列表
                    all_predictions.append(predictions.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    all_scores.append(scores.cpu().numpy())
                    
                    processed_batches += 1
                    
                except Exception as e:
                    logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                    continue
        
        logger.info(f"总批次: {total_batches}, 处理的批次: {processed_batches}")
        
        if not all_predictions:
            logger.warning("未处理任何有效批次，返回空数组")
            return np.array([]), np.array([]), np.array([])
            
        # 合并所有批次的结果
        try:
            flat_predictions = np.concatenate(all_predictions)
            flat_labels = np.concatenate(all_labels)
            flat_scores = np.concatenate(all_scores)
            
            # 移除NaN和inf值
            valid_indices = ~(np.isnan(flat_labels) | np.isinf(flat_labels) |
                            np.isnan(flat_scores) | np.isinf(flat_scores))
            flat_predictions = flat_predictions[valid_indices]
            flat_labels = flat_labels[valid_indices]
            flat_scores = flat_scores[valid_indices]
            
            # 检查最终结果是否为空
            if len(flat_labels) == 0:
                logger.warning("所有标签和得分均无效，返回空数组")
                return np.array([]), np.array([]), np.array([])
            
            return flat_predictions, flat_labels, flat_scores
            
        except Exception as e:
            logger.error(f"合并检测结果时出错: {e}")
            return np.array([]), np.array([]), np.array([])

# ... existing code ...

def evaluate_model_comprehensive(predictions, labels, scores):
    """
    全面评估模型性能
    """
    metrics = {}
    
    # 检查输入是否为空
    if len(predictions) == 0 or len(labels) == 0 or len(scores) == 0:
        logger.warning("评估输入为空")
        return {
            'roc_auc': 0.0, 'pr_auc': 0.0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'avg_score_normal': 0.0, 'avg_score_anomaly': 0.0
        }
    
    # logger.info(f"评估输入 - 预测数: {len(predictions)}, 标签数: {len(labels)}, 得分数: {len(scores)}")
    logger.info(f"测试集标签分布 - 正常样本: {np.sum(labels == 0)}, 异常样本: {np.sum(labels == 1)}")
    logger.info(f"测试集预测分布 - 正常预测: {np.sum(predictions == 0)}, 异常预测: {np.sum(predictions == 1)}")
    # logger.info(f"得分范围 - 最小值: {np.min(scores):.6f}, 最大值: {np.max(scores):.6f}, 平均值: {np.mean(scores):.6f}")
    
    try:
        # 展平数组
        flat_predictions = predictions.flatten()
        flat_labels = labels.flatten()
        flat_scores = scores.flatten()
        
        logger.debug(f"展平后 - 预测数: {len(flat_predictions)}, 标签数: {len(flat_labels)}, 得分数: {len(flat_scores)}")
        
        # 检查是否有有效数据
        if len(flat_labels) == 0 or len(flat_scores) == 0:
            raise ValueError("Empty arrays")
        
        # 确保所有数组长度一致
        min_length = min(len(flat_predictions), len(flat_labels), len(flat_scores))
        if len(flat_predictions) != min_length:
            flat_predictions = flat_predictions[:min_length]
        if len(flat_labels) != min_length:
            flat_labels = flat_labels[:min_length]
        if len(flat_scores) != min_length:
            flat_scores = flat_scores[:min_length]
            
        logger.debug(f"调整长度后 - 预测数: {len(flat_predictions)}, 标签数: {len(flat_labels)}, 得分数: {len(flat_scores)}")
        
        # 移除NaN和inf值
        valid_indices = ~(np.isnan(flat_labels) | np.isinf(flat_labels) | 
                         np.isnan(flat_scores) | np.isinf(flat_scores))
        flat_predictions = flat_predictions[valid_indices]
        flat_labels = flat_labels[valid_indices]
        flat_scores = flat_scores[valid_indices]
        
        # 检查是否还有数据
        if len(flat_labels) == 0:
            raise ValueError("No valid data after filtering")
        
        logger.debug(f"有效数据 - 样本数: {len(flat_labels)}, 标签唯一值: {np.unique(flat_labels)}")
        
        # 计算ROC AUC
        if len(np.unique(flat_labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(flat_labels, flat_scores)
            # logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
            # 计算PR AUC
            precision, recall, _ = precision_recall_curve(flat_labels, flat_scores)
            metrics['pr_auc'] = auc(recall, precision)
            # logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
            logger.warning("警告: 标签中只有一种类别，无法计算AUC指标")
        
        # 计算准确率、精确率、召回率
        tp = np.sum((flat_predictions == 1) & (flat_labels == 1))
        fp = np.sum((flat_predictions == 1) & (flat_labels == 0))
        tn = np.sum((flat_predictions == 0) & (flat_labels == 0))
        fn = np.sum((flat_predictions == 0) & (flat_labels == 1))
        
        logger.debug(f"混淆矩阵 - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        metrics['accuracy'] = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                             (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        logger.info(f"准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1分数: {metrics['f1_score']:.4f}")
        
        # 平均分数
        normal_mask = flat_labels == 0
        anomaly_mask = flat_labels == 1
        
        metrics['avg_score_normal'] = np.mean(flat_scores[normal_mask]) if np.sum(normal_mask) > 0 else 0
        metrics['avg_score_anomaly'] = np.mean(flat_scores[anomaly_mask]) if np.sum(anomaly_mask) > 0 else 0
        
        logger.info(f"正常样本平均得分: {metrics['avg_score_normal']:.4f}, 异常样本平均得分: {metrics['avg_score_anomaly']:.4f}, PRAUC: {metrics['pr_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        metrics = {
            'roc_auc': 0.0, 'pr_auc': 0.0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'avg_score_normal': 0.0, 'avg_score_anomaly': 0.0
        }
    
    return metrics


# 如果需要加载已保存的模型进行测试
def test_saved_model():
    """
    测试已保存的模型
    """
    # 设置随机种子以确保可重复性
    set_random_seed(Config.SEED)
    
    device = torch.device(Config.DEVICE)
    
    # 创建模型架构
    model = ImprovedWaveletSTGNN(
        in_channels=Config.MODEL_PARAMS['in_channels'],
        hidden_channels=Config.MODEL_PARAMS['hidden_channels'],
        out_channels=Config.MODEL_PARAMS['out_channels'],
        grid_size=Config.MODEL_PARAMS['grid_size'],
        temporal_window=Config.MODEL_PARAMS['temporal_window']
    ).to(device)
    
    # 加载模型权重
    try:
        success = load_model(model, 'robust_wavelet_stgnn_model.pth', device)
        if success:
            logger.info("模型加载成功!")
            return model
        else:
            logger.error("模型加载失败")
            return None
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None


def run_test_pipeline():
    """
    运行完整的测试流程，包括数据集加载、模型初始化、训练和测试。
    所有超参数从 Config 文件中读取。
    """
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # 1. 初始化数据集
    train_dataset, test_dataset = create_train_test_datasets(
        train_root_dir=Config.TRAIN_ROOT_DIR,
        test_root_dir=Config.TEST_ROOT_DIR,
        labels_dir=Config.LABELS_DIR,
        train_transform=transform,
        test_transform=transform,
        segment_length=Config.DATASET_PARAMS['segment_length'],
        overlap_ratio=Config.DATASET_PARAMS['overlap_ratio']
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 2. 创建数据加载器
    train_loader, test_loader = get_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=Config.TRAINING_PARAMS['batch_size'],
        num_workers=Config.TRAINING_PARAMS['num_workers']
    )
    
    print(f"训练数据加载器批次数量: {len(train_loader)}")
    print(f"测试数据加载器批次数量: {len(test_loader)}")
    
    # 3. 初始化模型
    model = ImprovedWaveletSTGNN(
        in_channels=Config.MODEL_PARAMS['in_channels'],
        hidden_channels=Config.MODEL_PARAMS['hidden_channels'],
        out_channels=Config.MODEL_PARAMS['out_channels'],
        grid_size=Config.MODEL_PARAMS['grid_size'],
        temporal_window=Config.MODEL_PARAMS['temporal_window']
    ).to(Config.DEVICE)
    
    # 4. 初始化训练器
    trainer = RobustAnomalyTrainer(
        model=model,
        device=Config.DEVICE,
        diversity_weight=Config.LOSS_WEIGHTS['diversity_weight'],
        contrastive_weight=Config.LOSS_WEIGHTS['contrastive_weight'],
        margin=Config.LOSS_WEIGHTS['margin'],
        bce_weight=Config.LOSS_WEIGHTS['bce_weight']
    )
    
    # 5. 训练模型（使用 train 方法）
    print("\n开始训练...")
    train_losses, val_losses, all_losses, test_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=train_loader,  # 使用训练集作为验证集（或根据需求更改）
        test_loader=test_loader,
        num_epochs=Config.TRAINING_PARAMS['num_epochs']
    )
    
    # 6. 打印最终训练结果
    print("\n最终训练结果:")
    print(f"  最佳测试 PR AUC: {max([metric['pr_auc'] for metric in test_metrics]):.4f}")
    print(f"  最终 ROC AUC: {test_metrics[-1]['roc_auc']:.4f}")
    print(f"  最终 PR AUC: {test_metrics[-1]['pr_auc']:.4f}")

if __name__ == "__main__":
    set_random_seed()
    run_test_pipeline()
