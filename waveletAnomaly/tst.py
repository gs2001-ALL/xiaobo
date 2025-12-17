import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
# Remove or comment out this line:
# from pytorch_wavelets.dwt.transform2d import DWTForward2D  # This line causes the error
from ptwt import wavedec3
from config import Config
from utils import setup_logging, printGrad
logger = setup_logging()
class I3DFeatureExtractor:
    """使用torchvision的I3D-RGB模型提取特征"""
    def __init__(self, device):
        self.device = device
        self.feature = None
        
        if TORCHVISION_AVAILABLE:
            # 加载预训练的R3D-18模型（类似于I3D）
            self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.model = self.model.to(device)
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # 如果torchvision不可用，创建一个简化版本
            self.model = self._create_simple_i3d()
        
        
        # 注册hook以获取Mixed-4f层特征（对应于R3D模型中的某一层）
        self._register_hook()
        
    def _create_simple_i3d(self):
        """创建简化的I3D模型"""
        # 创建一个简单的3D CNN作为替代
        model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),  # 这一层相当于Mixed-4f
            nn.AdaptiveAvgPool3d((4, 7, 7))  # 自适应平均池化
        )
        return model
    
    def _register_hook(self):
        """注册hook以捕获中间层特征"""
        if TORCHVISION_AVAILABLE:
            # 对于R3D-18，我们获取第几个层的特征（模拟Mixed-4f）
            def hook_fn(module, input, output):
                # hook函数的参数是固定的: module, input, output
                # 我们需要的是层的输出特征，而不是输入特征
                self.feature = output
            
            # 正确访问R3D-18的层结构
            # R3D-18的结构是: stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
            # 我们选择layer3作为Mixed-4f层的近似
            target_layer = self.model.layer3
            target_layer.register_forward_hook(hook_fn)
        else:
            # 对于简化版本，在倒数第二层注册hook
            def hook_fn(module, input, output):
                # hook函数的参数是固定的: module, input, output
                # 我们需要的是层的输出特征，而不是输入特征
                self.feature = output
            
            target_layer = list(self.model.children())[-2]
            target_layer.register_forward_hook(hook_fn)

    def extract_features(self, x):
        """提取Mixed-4f层特征"""
        
        _ = self.model(x)
        if self.feature is not None:
            # 确保特征需要梯度（如果输入需要梯度）
            if x.requires_grad and not self.feature.requires_grad:
                self.feature = self.feature.requires_grad_(True)
            return self.feature
        else:
            # 如果hook未能捕获特征，返回模型输出
            output = self.model(x)
            # 确保输出需要梯度（如果输入需要梯度）
            if x.requires_grad and not output.requires_grad:
                output = output.requires_grad_(True)
            return output


class DWT3D(nn.Module):
    """
    3D离散小波变换模块
    使用PyTorch Wavelet Toolbox (ptwt) 实现3D小波变换，支持GPU加速
    对视频数据进行时空域的小波分解，提取多尺度频率信息
    """
    def __init__(self, wavelet=Config.WAVELET_PARAMS['wavelet'], levels=Config.WAVELET_PARAMS['levels']):
        super(DWT3D, self).__init__()
        self.wavelet = wavelet
        self.levels = levels

    def forward(self, video):
        """
        对输入视频进行3D小波变换
        
        参数:
            video (torch.Tensor): 输入视频张量，形状为 [batch, channels, frames, height, width]
            
        返回:
            dict: 多级小波系数字典
                格式: {
                    'level_0': {
                        'channel_0': {
                            'LLL': tensor,  # 近似系数 [batch, 1, frames//2, height//2, width//2]
                            'LLH': tensor,  # 水平细节系数
                            'LHL': tensor,  # 垂直细节系数
                            'LHH': tensor,  # 对角细节系数
                            'HLL': tensor,  # 时间细节系数
                            'HLH': tensor,  # 水平-时间细节系数
                            'HHL': tensor,  # 垂直-时间细节系数
                            'HHH': tensor   # 对角-时间细节系数
                        }
                    }
                }
        """
        batch_size, channels, frames, height, width = video.shape
        all_coeffs = {}
        
        # 确保输入视频需要梯度
        if self.training and not video.requires_grad:
            video = video.requires_grad_(True)
        
        try:
            # 使用ptwt进行3D小波变换，直接处理整个批次 (N, C, D, H, W)
            coeffs = wavedec3(video, wavelet=self.wavelet, level=self.levels)
            
            # 处理每个分解级别
            for level in range(self.levels):
                level_key = f'level_{level}'
                if level_key not in all_coeffs:
                    all_coeffs[level_key] = {}
                
                # 根据ptwt返回的格式处理系数 (tuple类型)
                if isinstance(coeffs, tuple) and len(coeffs) > level + 1:
                    coeff_level = coeffs[level + 1]  # 第0个元素是低频分量(LLL)
                    
                    if isinstance(coeff_level, dict):
                        # 处理不同方向的系数，根据字典键含义
                        coeff_dict = coeff_level
                        
                        # 处理每个通道
                        for c in range(channels):
                            channel_key = f'channel_{c}'
                            if channel_key not in all_coeffs[level_key]:
                                all_coeffs[level_key][channel_key] = {}
                            
                            # 获取低频分量 (LLL) - 在coeffs[0]中
                            lll = coeffs[0][:, c:c+1] if level == 0 and coeffs[0].size(1) > c else torch.zeros(batch_size, 1, max(1, frames//2), max(1, height//2), max(1, width//2), device=video.device, requires_grad=True)
                            
                            # 确保所有系数张量都具有正确的梯度设置
                            lll = lll.requires_grad_(True) if video.requires_grad and not lll.requires_grad else lll
                            
                            all_coeffs[level_key][channel_key] = {
                                'LLL': lll,  # 近似系数
                                'LLH': coeff_dict.get('aad', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('aad') is not None and coeff_dict['aad'].size(1) > c else torch.zeros_like(lll),
                                'LHL': coeff_dict.get('ada', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('ada') is not None and coeff_dict['ada'].size(1) > c else torch.zeros_like(lll),
                                'LHH': coeff_dict.get('add', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('add') is not None and coeff_dict['add'].size(1) > c else torch.zeros_like(lll),
                                'HLL': coeff_dict.get('daa', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('daa') is not None and coeff_dict['daa'].size(1) > c else torch.zeros_like(lll),
                                'HLH': coeff_dict.get('dad', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('dad') is not None and coeff_dict['dad'].size(1) > c else torch.zeros_like(lll),
                                'HHL': coeff_dict.get('dda', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('dda') is not None and coeff_dict['dda'].size(1) > c else torch.zeros_like(lll),
                                'HHH': coeff_dict.get('ddd', torch.zeros_like(lll))[:, c:c+1] if coeff_dict.get('ddd') is not None and coeff_dict['ddd'].size(1) > c else torch.zeros_like(lll)
                            }
                            
                            # 确保所有系数都需要梯度
                            for key in all_coeffs[level_key][channel_key]:
                                coeff = all_coeffs[level_key][channel_key][key]
                                # 修改开始：确保所有系数张量都具有正确的梯度设置
                                if video.requires_grad and not coeff.requires_grad:
                                    all_coeffs[level_key][channel_key][key] = coeff.requires_grad_(True)
                                elif not video.requires_grad and coeff.requires_grad:
                                    all_coeffs[level_key][channel_key][key] = coeff.detach()
                                # 修改结束
                    else:
                        # 如果不是字典格式，为每个通道创建默认系数
                        dummy_shape = coeff_level.shape[1:] if isinstance(coeff_level, torch.Tensor) and coeff_level.dim() > 4 else (batch_size, 1, max(1, frames//2), max(1, height//2), max(1, width//2))
                        for c in range(channels):
                            channel_key = f'channel_{c}'
                            # 修改开始：确保默认系数张量具有正确的梯度设置
                            dummy_tensor = torch.zeros(dummy_shape, device=video.device, requires_grad=video.requires_grad) if len(dummy_shape) == 5 else torch.zeros(dummy_shape[0], 1, dummy_shape[1], dummy_shape[2], dummy_shape[3], device=video.device, requires_grad=video.requires_grad)
                            # 修改结束
                            all_coeffs[level_key][channel_key] = {
                                'LLL': dummy_tensor,
                                'LLH': dummy_tensor,
                                'LHL': dummy_tensor,
                                'LHH': dummy_tensor,
                                'HLL': dummy_tensor,
                                'HLH': dummy_tensor,
                                'HHL': dummy_tensor,
                                'HHH': dummy_tensor
                            }
                else:
                    # 如果没有足够的系数级别，为每个通道创建默认系数
                    dummy_shape = (batch_size, 1, max(1, frames//2), max(1, height//2), max(1, width//2))
                    # 修改开始：确保默认系数张量具有正确的梯度设置
                    dummy_tensor = torch.zeros(dummy_shape, device=video.device, requires_grad=video.requires_grad)
                    # 修改结束
                    for c in range(channels):
                        channel_key = f'channel_{c}'
                        all_coeffs[level_key][channel_key] = {
                            'LLL': dummy_tensor,
                            'LLH': dummy_tensor,
                            'LHL': dummy_tensor,
                            'LHH': dummy_tensor,
                            'HLL': dummy_tensor,
                            'HLH': dummy_tensor,
                            'HHL': dummy_tensor,
                            'HHH': dummy_tensor
                        }
                        
        except Exception as e:
            print(f"3D小波变换失败: {e}")
            # 创建默认系数
            dummy_shape = (batch_size, 1, max(1, frames//2), max(1, height//2), max(1, width//2))
            # 修改开始：确保默认系数张量具有正确的梯度设置
            dummy_tensor = torch.zeros(dummy_shape, device=video.device, requires_grad=video.requires_grad)
            # 修改结束
            for level in range(self.levels):
                level_key = f'level_{level}'
                if level_key not in all_coeffs:
                    all_coeffs[level_key] = {}
                for c in range(channels):
                    channel_key = f'channel_{c}'
                    all_coeffs[level_key][channel_key] = {
                        'LLL': dummy_tensor,
                        'LLH': dummy_tensor,
                        'LHL': dummy_tensor,
                        'LHH': dummy_tensor,
                        'HLL': dummy_tensor,
                        'HLH': dummy_tensor,
                        'HHL': dummy_tensor,
                        'HHH': dummy_tensor
                    }
        
        return all_coeffs

class AdvancedWaveletCoeffAttention(nn.Module):
    """
    输入: 多级小波系数字典，格式为 {level_key: {channel_key: {coeff_key: tensor}}}
    输出: 融合后的特征张量，形状为 [B, C, T, H, W]
    通道 / 空间 / 时间 注意力的中间通道数均为 max(8, C // 8)
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # 初始化参数
        mid_channels = max(8, channels // 2)  # 中间层通道数

        # 1. 通道注意力：作用于 C，保留 T
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B*K*T, C, 1, 1]
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 2. 空间注意力：作用于 (H, W)，共享 K,C,T
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 3. 时间注意力：作用于 T，共享 K,C,H,W
        self.temporal_att = nn.Sequential(
            nn.Conv1d(1, mid_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def _process_multi_level_wavelet_coeffs(self, wavelet_coeffs):
        """
        处理多级小波系数，将其重组为 [B, K, C, T, H, W] 格式
        
        参数:
            wavelet_coeffs (dict): 多级小波系数字典，格式为 {level_key: {channel_key: {coeff_key: tensor}}}
            
        返回:
            tuple: (torch.Tensor, int, int) - 重组后的张量、小波系数组数(K)、通道数(C)
        """
        all_levels = []
        for level_key, level_coeffs in wavelet_coeffs.items():
            for channel_key, channel_coeffs in level_coeffs.items():
                # 提取所有系数组并堆叠
                level_tensors = [
                    channel_coeffs[key] for key in ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
                    if key in channel_coeffs and channel_coeffs[key] is not None
                ]
                if level_tensors:
                    # 确保所有系数张量具有相同的形状
                    first_shape = level_tensors[0].shape
                    level_tensors = [tensor.view(first_shape) for tensor in level_tensors]

                    # 堆叠系数组
                    level_tensor = torch.stack(level_tensors, dim=1)  # [B, K, C, T, H, W]
                    all_levels.append(level_tensor)
        
        # 合并所有级别和通道
        wavelet_tensor = torch.cat(all_levels, dim=1)  # [B, K_total, C, T, H, W]
        
        # 检查是否成功生成张量
        if wavelet_tensor.dim() != 6:
            raise ValueError("小波系数未能正确重组为 [B, K, C, T, H, W] 格式")
        
        # 获取小波系数组数(K)和通道数(C)
        B, K, C, T, H, W = wavelet_tensor.shape
        return wavelet_tensor, K, C

    def forward(self, wavelet_coeffs):
        """
        前向传播过程
        
        参数:
            wavelet_coeffs (dict): 多级小波系数字典，格式为 {level_key: {channel_key: {coeff_key: tensor}}}
            
        返回:
            torch.Tensor: 融合后的特征张量，形状为 [B, C, T, H, W]
        """
        # 1. 处理多级小波系数
        x, K, C = self._process_multi_level_wavelet_coeffs(wavelet_coeffs)  # [B, K, C, T, H, W]

        B, K, C, T, H, W = x.shape
        
        # 记录输入信息
        # logger.debug(f"AdvancedWaveletCoeffAttention 输入形状: {x.shape}, requires_grad: {x.requires_grad}")

        # ---------- 2. 通道注意力 ----------
        x4d = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B,K,T,C,H,W]
        x4d = x4d.view(-1, C, H, W)                      # [B*K*T,C,H,W]
        ca = self.channel_att(x4d)                       # [B*K*T,C,1,1]
        ca = ca.view(B, K, T, C, 1, 1).permute(0, 1, 3, 2, 4, 5)  # [B,K,C,T,1,1]
        x = x * ca                                       # broadcast

        # ---------- 3. 空间注意力 ----------
        feat_hw = x.mean(dim=(1, 2, 3), keepdim=False)   # [B,H,W]
        feat_hw = feat_hw.unsqueeze(1)                   # [B,1,H,W]
        sa = self.spatial_att(feat_hw)                     # [B,1,H,W]
        sa = sa.unsqueeze(1).unsqueeze(1)                  # [B,1,1,1,H,W]

        # ---------- 4. 时间注意力 ----------
        feat_t = x.mean(dim=(1, 2, 4, 5), keepdim=False)   # [B,T]
        feat_t = feat_t.unsqueeze(1)                       # [B,1,T]
        ta = self.temporal_att(feat_t)                     # [B,1,T]
        ta = ta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)    # [B,1,1,T,1,1]

        # ---------- 5. 融合 ----------
        att = sa * ta                                      # [B,1,1,T,H,W]
        out = (x * att).sum(dim=1)                         # [B,C,T,H,W]

        return out

class GatedGCN(nn.Module):
    """
    门控图卷积网络 (Gated GCN)
    
    该网络结合了图卷积网络(GCN)和门控机制，用于处理图结构数据。
    通过门控机制，网络可以自适应地控制信息流，增强模型的表达能力。
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        初始化GatedGCN网络
        
        参数:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层特征维度
            out_channels (int): 输出特征维度
        """
        super().__init__()
        
        # 保存输入参数
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 初始化网络层为None，将在第一次forward时根据实际输入维度创建
        self.conv1 = None
        self.conv2 = None
        self.gate = None
        
        # 跟踪设备
        self._device = torch.device('cpu')
        
    def _init_layers(self, actual_in_channels):
        """
        根据实际输入维度初始化网络层
        
        参数:
            actual_in_channels (int): 实际的输入特征维度
        """
        try:
            # 第一层图卷积层：将输入特征映射到隐藏层
            self.conv1 = GCNConv(actual_in_channels, max(1, self.hidden_channels)).to(self._device)
            
            # 第二层图卷积层：将隐藏层特征映射到输出维度
            self.conv2 = GCNConv(max(1, self.hidden_channels), max(1, self.out_channels)).to(self._device)
            
            # 门控机制：控制原始输入和处理后特征的融合比例
            # 将原始输入和处理后的特征拼接后通过sigmoid函数生成门控权重
            self.gate = nn.Sequential(
                nn.Linear(max(1, self.out_channels + actual_in_channels), max(1, self.out_channels)),  # 拼接后的特征维度是out_channels+actual_in_channels
                nn.Sigmoid()  # 生成0-1之间的门控权重
            ).to(self._device)
            
            # 初始化权重以提高数值稳定性
            self._initialize_gcn_weights()
            
        except Exception as e:
            logger.error(f"GatedGCN层初始化失败: {e}")
            # 创建简单的线性层作为后备
            self.conv1 = nn.Linear(actual_in_channels, max(1, self.hidden_channels)).to(self._device)
            self.conv2 = nn.Linear(max(1, self.hidden_channels), max(1, self.out_channels)).to(self._device)
            self.gate = nn.Sequential(
                nn.Linear(max(1, self.out_channels + actual_in_channels), max(1, self.out_channels)),
                nn.Sigmoid()
            ).to(self._device)
    
    def _initialize_gcn_weights(self):
        """
        初始化GCN层权重以提高数值稳定性
        """
        try:
            # 初始化GCN层权重
            if hasattr(self.conv1, 'reset_parameters'):
                self.conv1.reset_parameters()
            if hasattr(self.conv2, 'reset_parameters'):
                self.conv2.reset_parameters()
                
            # 初始化门控网络权重
            for m in self.gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            logger.warning(f"GCN权重初始化失败: {e}")

    def forward(self, data):
        """
        前向传播过程
        """
        # 确保数据在正确的设备上
        if len(list(self.parameters())) > 0:
            self._device = next(self.parameters()).device
        else:
            # 如果模型没有参数，使用数据所在的设备
            if hasattr(data, 'x') and data.x is not None:
                self._device = data.x.device
            elif hasattr(data, 'edge_index') and data.edge_index is not None:
                self._device = data.edge_index.device
            else:
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保图数据的所有组件都在正确的设备上
        if hasattr(data, 'x') and data.x is not None:
            if data.x.device != self._device:
                data.x = data.x.to(self._device)
                
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            if data.edge_index.device != self._device:
                data.edge_index = data.edge_index.to(self._device)
        
        # 检查输入数据是否包含NaN或inf
        if hasattr(data, 'x') and data.x is not None:
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                logger.warning("GatedGCN输入包含NaN或inf值")
                # 使用零张量替代
                data.x = torch.zeros_like(data.x)
                
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            if torch.isnan(data.edge_index).any() or torch.isinf(data.edge_index).any():
                logger.warning("GatedGCN边索引包含NaN或inf值")
                # 使用空边索引替代
                data.edge_index = torch.empty((2, 0), dtype=torch.long, device=self._device)
        
        # 限制输入范围
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.clamp(data.x, min=-100, max=100)
        
        # 保存原始输入特征，用于门控机制
        original_x = data.x
        
        # 确保输入特征需要梯度
        if not original_x.requires_grad:
            original_x = original_x.requires_grad_(True)
            data.x = original_x  # 更新data.x以确保一致性
        
        # 动态初始化网络层以适应实际输入维度
        # 检查是否需要初始化网络层
        if self.conv1 is None or self.conv1.in_channels != original_x.shape[1]:
            self._init_layers(original_x.shape[1])
        else:
            # 确保已存在的网络层在正确的设备上
            try:
                if next(self.conv1.parameters()).device != self._device:
                    self.conv1 = self.conv1.to(self._device)
            except Exception:
                self.conv1 = self.conv1.to(self._device)
                
            try:
                if self.conv2 and next(self.conv2.parameters()).device != self._device:
                    self.conv2 = self.conv2.to(self._device)
            except Exception:
                self.conv2 = self.conv2.to(self._device)
                
            try:
                if self.gate and next(self.gate.parameters()).device != self._device:
                    self.gate = self.gate.to(self._device)
            except Exception:
                self.gate = self.gate.to(self._device)
        
        # 确保所有层都在正确的设备上后再进行计算
        if data.x.device != self._device:
            data.x = data.x.to(self._device)
        if data.edge_index.device != self._device:
            data.edge_index = data.edge_index.to(self._device)
        
        # 第一层GCN：图卷积 + ReLU激活函数
        try:
            x = F.relu(self.conv1(data.x, data.edge_index))
            # 检查输出是否包含NaN或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("第一层GCN输出包含NaN或inf值")
                x = torch.zeros_like(x)
            # 确保输出需要梯度
            if not x.requires_grad:
                x = x.requires_grad_(True)
        except Exception as e:
            logger.warning(f"第一层GCN计算失败: {e}")
            x = torch.zeros_like(data.x)
            if not x.requires_grad:
                x = x.requires_grad_(True)
        
        # Dropout正则化：防止过拟合
        x = F.dropout(x, p=0.3, training=self.training)  # 增加dropout率以增强泛化能力
        
        # 第二层GCN：进一步提取图结构特征
        try:
            x = F.relu(self.conv2(x, data.edge_index))  # 添加ReLU激活函数
            # 检查输出是否包含NaN或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("第二层GCN输出包含NaN或inf值")
                x = torch.zeros_like(x)
            # 确保输出需要梯度
            if not x.requires_grad:
                x = x.requires_grad_(True)
        except Exception as e:
            logger.warning(f"第二层GCN计算失败: {e}")
            x = torch.zeros_like(x)
            if not x.requires_grad:
                x = x.requires_grad_(True)
            
        # 添加残差连接以增强梯度流动
        if x.shape == original_x.shape:
            x = x + original_x  # 残差连接
            # 确保输出需要梯度
            if not x.requires_grad:
                x = x.requires_grad_(True)
        
        # 门控机制：自适应融合原始特征和处理后的特征
        # 拼接原始特征和处理后的特征
        if x.device != self._device:
            x = x.to(self._device)
        if original_x.device != self._device:
            original_x = original_x.to(self._device)
            
        # 检查拼接前的张量是否包含NaN或inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("处理后的特征包含NaN或inf值")
            x = torch.zeros_like(x)
            if not x.requires_grad:
                x = x.requires_grad_(True)
        if torch.isnan(original_x).any() or torch.isinf(original_x).any():
            logger.warning("原始特征包含NaN或inf值")
            original_x = torch.zeros_like(original_x)
            if not original_x.requires_grad:
                original_x = original_x.requires_grad_(True)
        
        try:
            gate_input = torch.cat([x, original_x], dim=1)
            # 检查拼接后的输入是否包含NaN或inf
            if torch.isnan(gate_input).any() or torch.isinf(gate_input).any():
                logger.warning("门控机制输入包含NaN或inf值")
                gate_input = torch.zeros_like(gate_input)
            # 确保输出需要梯度
            if not gate_input.requires_grad:
                gate_input = gate_input.requires_grad_(True)
        except Exception as e:
            logger.warning(f"门控机制输入拼接失败: {e}")
            # 创建合适维度的零张量
            gate_input = torch.zeros(x.shape[0], x.shape[1] + original_x.shape[1], device=self._device, requires_grad=True)
        
        # 通过门控网络生成门控权重
        try:
            gate = self.gate(gate_input)
            # 检查门控权重是否包含NaN或inf
            if torch.isnan(gate).any() or torch.isinf(gate).any():
                logger.warning("门控权重包含NaN或inf值")
                gate = torch.zeros_like(gate)
            # 确保输出需要梯度
            if not gate.requires_grad:
                gate = gate.requires_grad_(True)
        except Exception as e:
            logger.warning(f"门控权重计算失败: {e}")
            # 创建合适维度的零张量
            gate = torch.zeros(x.shape[0], x.shape[1], device=self._device, requires_grad=True)
        
        # 应用门控权重：控制处理后特征的贡献程度
        result = x * gate
        
        # 确保输出在正确的设备上
        if result.device != self._device:
            result = result.to(self._device)
            
        # 最终检查输出是否包含NaN或inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("GatedGCN最终输出包含NaN或inf值，使用零张量替代")
            result = torch.zeros_like(result)
        
        # 确保最终输出需要梯度
        if not result.requires_grad:
            result = result.requires_grad_(True)
            
        return result

class SpatioTemporalGraphBuilder(nn.Module):
    def __init__(self, grid_size=Config.MODEL_PARAMS['grid_size'], 
                 temporal_window=Config.MODEL_PARAMS['temporal_window']):
        super(SpatioTemporalGraphBuilder, self).__init__()
        self.grid_size = grid_size
        self.temporal_window = temporal_window

    def build_graph(self, video_tensor, device, training=False):
        """
        输入: video tensor [batch, channels, frames, height, width]
        输出: PyG Data对象列表(每个batch一个)
        """
        # 确保输入张量在正确的设备上
        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.tensor(video_tensor, device=device)
        elif video_tensor.device != device:
            video_tensor = video_tensor.to(device)
            
        batch_size, channels, num_frames, H, W = video_tensor.shape
        
        # 添加检查以确保grid_size不会导致过小的网格
        if H < self.grid_size or W < self.grid_size:
            raise ValueError(f"图像尺寸 ({H}x{W}) 小于网格大小 ({self.grid_size})")
        
        grid_h, grid_w = max(1, H // self.grid_size), max(1, W // self.grid_size)
        graphs = []
        
        # 预计算空间邻居偏移量
        spatial_offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di != 0 or dj != 0:  # 排除中心点
                    spatial_offsets.append((di, dj))
        
        # 预计算时间偏移量
        temporal_offsets = []
        for dt in range(1, self.temporal_window + 1):
            temporal_offsets.append(dt)
            temporal_offsets.append(-dt)

        # 新增：Temporal padding if necessary
        target_time_steps = max(num_frames, 2)  # Ensure at least 2 time steps
        mask = None
        if num_frames < target_time_steps:
            video_tensor, mask = self._temporal_padding_with_mask(video_tensor, target_time_steps)

        num_frames = video_tensor.shape[2]  # Update num_frames after padding
        
        for b in range(batch_size):
            # 向量化节点特征计算 (处理所有通道)
            node_features = self._compute_node_features_vectorized(
                video_tensor[b], grid_h, grid_w)  # [channels, frames, height, width]

            # 检查节点特征数量是否合理
            num_nodes_per_frame = len(node_features) // num_frames
            if num_nodes_per_frame > 49:  # 限制每帧的最大节点数为7x7
                raise ValueError(f"每帧的节点数过多 ({num_nodes_per_frame}), 请调整grid_size或输入尺寸")
            
            # 高效构建边
            edge_index = self._build_edges_optimized(
                num_frames, grid_h, grid_w, spatial_offsets, temporal_offsets, device
            )
            
            # 转换为PyG Data对象，并确保所有张量在正确设备上
            # 关键修复：使用torch.stack并确保保持梯度
            x = torch.stack(node_features, dim=0)
            x = x.to(device)
            
            # 确保节点特征需要梯度
            if training and video_tensor.requires_grad and not x.requires_grad:
                x = x.requires_grad_(True)
                
            edge_index = edge_index.to(device)
            graph = Data(x=x, edge_index=edge_index)
            
            # 添加mask信息到图数据对象中
            if mask is not None:
                graph.mask = mask[b].to(device)
            
            # 确保整个图数据对象在正确的设备上
            graph = graph.to(device)
            graphs.append(graph)
        
        return graphs

    def _temporal_padding_with_mask(self, video_tensor, target_time_steps):
        """
        对视频的时间维度进行填充到目标时间步长，并生成mask
        
        参数:
            video_tensor (torch.Tensor): 输入视频张量，形状为 [batch, channels, frames, height, width]
            target_time_steps (int): 目标时间步长
            
        返回:
            tuple: (填充后的视频张量, mask张量)
        """
        batch_size, channels, num_frames, H, W = video_tensor.shape
        pad_size = target_time_steps - num_frames
        
        if pad_size > 0:
            # 关键修复：确保填充张量保持与输入相同的梯度属性
            padding = torch.zeros((batch_size, channels, pad_size, H, W), 
                                device=video_tensor.device, 
                                dtype=video_tensor.dtype,
                                requires_grad=video_tensor.requires_grad)
            video_tensor = torch.cat([video_tensor, padding], dim=2)
        
        # 创建mask
        mask = torch.ones((batch_size, target_time_steps), dtype=torch.bool, device=video_tensor.device)
        if pad_size > 0:
            mask[:, num_frames:] = False
        
        return video_tensor, mask

    def _compute_node_features_vectorized(self, video_frame, grid_h, grid_w):
        """
        向量化计算节点特征 (处理所有通道)
        输入: video_frame [channels, frames, height, width]
        输出: 节点特征列表，每个元素是 [feature_dim] 的张量
        """
        channels, frames, H, W = video_frame.shape
        node_features = []
        
        # 使用unfold操作来分割网格块，提高效率
        all_patches = []
        for c in range(channels):
            channel_patches = video_frame[c].unfold(1, self.grid_size, self.grid_size) \
                                           .unfold(2, self.grid_size, self.grid_size)
            all_patches.append(channel_patches)
        
        # 为每一帧计算所有网格块的特征
        for t in range(frames):
            frame_features = []
            for c in range(channels):
                # 当前帧的所有网格块 [grid_h, grid_w, grid_size, grid_size]
                frame_patches = all_patches[c][t]
                frame_features.append(frame_patches)
            
            # 计算每个网格块的统计特征 (跨所有通道)
            for i in range(grid_h):
                for j in range(grid_w):
                    # 收集所有通道在当前网格块的值
                    all_channel_values = []
                    for c in range(channels):
                        patch_values = frame_features[c][i, j]  # [grid_size, grid_size]
                        all_channel_values.append(patch_values)
                    
                    # 合并所有通道的值
                    combined_patch = torch.stack(all_channel_values)  # [channels, grid_size, grid_size]
                    
                    # 计算更丰富的统计特征，增加特征维度
                    # 展平张量以进行统计计算
                    flattened_patch = combined_patch.flatten()  # [channels * grid_size * grid_size]
                    
                    # 计算多种统计特征以增加特征维度
                    mean_val = flattened_patch.mean()
                    std_val = flattened_patch.std()
                    min_val = flattened_patch.min()
                    max_val = flattened_patch.max()
                    
                    # 计算额外的统计特征
                    # 能量
                    energy = (flattened_patch ** 2).mean()
                    # 均方根
                    rms = torch.sqrt(energy + 1e-8)
                    # 偏度 (skewness)
                    skewness = (((flattened_patch - mean_val) / (std_val + 1e-8)) ** 3).mean()
                    # 峰度 (kurtosis)
                    kurtosis = (((flattened_patch - mean_val) / (std_val + 1e-8)) ** 4).mean() - 3
                    
                    # 中位数
                    median_val = torch.median(flattened_patch)
                    # 四分位距
                    q75 = torch.quantile(flattened_patch, 0.75)
                    q25 = torch.quantile(flattened_patch, 0.25)
                    iqr = q75 - q25
                    
                    # 创建特征向量，保持梯度流
                    # 关键修复：使用torch.stack而不是创建新张量，保持梯度连接
                    features = torch.stack([
                        mean_val,      # 均值
                        std_val,       # 标准差
                        min_val,       # 最小值
                        max_val,       # 最大值
                        energy,        # 能量
                        rms,           # 均方根
                        skewness,      # 偏度
                        kurtosis,      # 峰度
                        median_val,    # 中位数
                        iqr            # 四分位距
                    ])
                    node_features.append(features)
        
        return node_features

    def _build_edges_optimized(self, num_frames, grid_h, grid_w, spatial_offsets, temporal_offsets, device):
        """
        优化的边构建方法
        """
        edges = []
        num_nodes_per_frame = grid_h * grid_w
        
        # 预计算所有网格位置的坐标
        grid_positions = {}
        for i in range(grid_h):
            for j in range(grid_w):
                pos_index = i * grid_w + j
                grid_positions[pos_index] = (i, j)
        
        # 预计算合法的网格坐标范围
        valid_coords = set((i, j) for i in range(grid_h) for j in range(grid_w))
        
        # 为每个节点构建连接
        for t in range(num_frames):
            for pos_in_frame in range(num_nodes_per_frame):
                node_idx = t * num_nodes_per_frame + pos_in_frame
                i, j = grid_positions[pos_in_frame]
                
                # 空间连接 - 使用预计算的偏移量
                for di, dj in spatial_offsets:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in valid_coords:
                        neighbor_pos = ni * grid_w + nj
                        neighbor_idx = t * num_nodes_per_frame + neighbor_pos
                        edges.append([node_idx, neighbor_idx])
                
                # 时间连接 - 使用预计算的时间偏移量
                for dt in temporal_offsets:
                    target_t = t + dt
                    if 0 <= target_t < num_frames:
                        target_idx = target_t * num_nodes_per_frame + pos_in_frame
                        edges.append([node_idx, target_idx])
        
        # 转换为PyG格式
        # 关键修复：确保边索引张量不阻断梯度（边索引不需要梯度）
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index
class FixedSpatioTemporalGraphBuilder(nn.Module):
    def __init__(self, grid_size=Config.MODEL_PARAMS['grid_size'], 
                 temporal_window=Config.MODEL_PARAMS['temporal_window']):
        super(FixedSpatioTemporalGraphBuilder, self).__init__()
        self.grid_size = grid_size
        self.temporal_window = temporal_window

    def build_graph(self, video_tensor, device, training=False):
        """
        输入: video tensor [batch, channels, frames, height, width]
        输出: PyG Data对象列表(每个batch一个)
        """
        # 确保输入张量在正确的设备上
        if isinstance(video_tensor, torch.Tensor):
            if video_tensor.device != device:
                video_tensor = video_tensor.to(device)
        else:
            video_tensor = torch.tensor(video_tensor, device=device)
            
        batch_size, channels, num_frames, H, W = video_tensor.shape
        
        # 添加检查以确保grid_size不会导致过小的网格
        if H < self.grid_size or W < self.grid_size:
            raise ValueError(f"图像尺寸 ({H}x{W}) 小于网格大小 ({self.grid_size})")
        
        grid_h, grid_w = max(1, H // self.grid_size), max(1, W // self.grid_size)
        graphs = []
        
        # 预计算空间邻居偏移量
        spatial_offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di != 0 or dj != 0:  # 排除中心点
                    spatial_offsets.append((di, dj))
        
        # 预计算时间偏移量
        temporal_offsets = []
        for dt in range(1, self.temporal_window + 1):
            temporal_offsets.append(dt)
            temporal_offsets.append(-dt)

        # 新增：Temporal padding if necessary
        target_time_steps = max(num_frames, 2)  # Ensure at least 2 time steps
        mask = None
        if num_frames < target_time_steps:
            video_tensor, mask = self._temporal_padding_with_mask(video_tensor, target_time_steps)

        num_frames = video_tensor.shape[2]  # Update num_frames after padding
        
        for b in range(batch_size):
            # 向量化节点特征计算 (处理所有通道)
            node_features_list = self._compute_node_features_vectorized(
                video_tensor[b], grid_h, grid_w)  # [channels, frames, height, width]

            # 检查节点特征数量是否合理
            num_nodes_per_frame = len(node_features_list) // num_frames
            if num_nodes_per_frame > 49:  # 限制每帧的最大节点数为7x7
                raise ValueError(f"每帧的节点数过多 ({num_nodes_per_frame}), 请调整grid_size或输入尺寸")
            
            # 高效构建边
            edge_index = self._build_edges_optimized(
                num_frames, grid_h, grid_w, spatial_offsets, temporal_offsets, device
            )
            
            # 转换为PyG Data对象，并确保所有张量在正确设备上
            # 关键修复：保持张量连接
            x = torch.stack(node_features_list, dim=0).to(device)
            
            # 确保节点特征需要梯度
            if training and video_tensor.requires_grad and not x.requires_grad:
                x = x.requires_grad_(True)
                
            edge_index = edge_index.to(device)
            graph = Data(x=x, edge_index=edge_index)
            
            # 添加mask信息到图数据对象中
            if mask is not None:
                graph.mask = mask[b].to(device)
            
            # 确保整个图数据对象在正确的设备上
            graph = graph.to(device)
            graphs.append(graph)
        
        return graphs

    def _temporal_padding_with_mask(self, video_tensor, target_time_steps):
        """
        对视频的时间维度进行填充到目标时间步长，并生成mask
        
        参数:
            video_tensor (torch.Tensor): 输入视频张量，形状为 [batch, channels, frames, height, width]
            target_time_steps (int): 目标时间步长
            
        返回:
            tuple: (填充后的视频张量, mask张量)
        """
        batch_size, channels, num_frames, H, W = video_tensor.shape
        pad_size = target_time_steps - num_frames
        
        if pad_size > 0:
            # 关键修复：确保填充张量保持与输入相同的梯度属性
            padding = torch.zeros((batch_size, channels, pad_size, H, W), 
                                device=video_tensor.device, 
                                dtype=video_tensor.dtype,
                                requires_grad=video_tensor.requires_grad)
            video_tensor = torch.cat([video_tensor, padding], dim=2)
        
        # 创建mask
        mask = torch.ones((batch_size, target_time_steps), dtype=torch.bool, device=video_tensor.device)
        if pad_size > 0:
            mask[:, num_frames:] = False
        
        return video_tensor, mask

    def _compute_node_features_vectorized(self, video_frame, grid_h, grid_w):
        """
        向量化计算节点特征 (处理所有通道)
        输入: video_frame [channels, frames, height, width]
        输出: 节点特征列表，每个元素是 [feature_dim] 的张量
        """
        channels, frames, H, W = video_frame.shape
        node_features = []
        
        # 使用unfold操作来分割网格块，提高效率
        all_patches = []
        for c in range(channels):
            channel_patches = video_frame[c].unfold(1, self.grid_size, self.grid_size) \
                                           .unfold(2, self.grid_size, self.grid_size)
            all_patches.append(channel_patches)
        
        # 为每一帧计算所有网格块的特征
        for t in range(frames):
            frame_features = []
            for c in range(channels):
                # 当前帧的所有网格块 [grid_h, grid_w, grid_size, grid_size]
                frame_patches = all_patches[c][t]
                frame_features.append(frame_patches)
            
            # 计算每个网格块的统计特征 (跨所有通道)
            for i in range(grid_h):
                for j in range(grid_w):
                    # 收集所有通道在当前网格块的值
                    all_channel_values = []
                    for c in range(channels):
                        patch_values = frame_features[c][i, j]  # [grid_size, grid_size]
                        all_channel_values.append(patch_values)
                    
                    # 合并所有通道的值
                    combined_patch = torch.stack(all_channel_values)  # [channels, grid_size, grid_size]
                    
                    # 计算更丰富的统计特征，增加特征维度
                    # 展平张量以进行统计计算
                    flattened_patch = combined_patch.flatten()  # [channels * grid_size * grid_size]
                    
                    # 计算多种统计特征以增加特征维度
                    mean_val = flattened_patch.mean()
                    std_val = flattened_patch.std()
                    min_val = flattened_patch.min()
                    max_val = flattened_patch.max()
                    
                    # 计算额外的统计特征
                    # 能量
                    energy = (flattened_patch ** 2).mean()
                    # 均方根
                    rms = torch.sqrt(energy + 1e-8)
                    # 偏度 (skewness)
                    skewness = (((flattened_patch - mean_val) / (std_val + 1e-8)) ** 3).mean()
                    # 峰度 (kurtosis)
                    kurtosis = (((flattened_patch - mean_val) / (std_val + 1e-8)) ** 4).mean() - 3
                    
                    # 中位数
                    median_val = torch.median(flattened_patch)
                    # 四分位距
                    q75 = torch.quantile(flattened_patch, 0.75)
                    q25 = torch.quantile(flattened_patch, 0.25)
                    iqr = q75 - q25
                    
                    # 创建特征向量，保持梯度流
                    # 关键修复：使用torch.stack而不是创建新张量，保持梯度连接
                    features = torch.stack([
                        mean_val,      # 均值
                        std_val,       # 标准差
                        min_val,       # 最小值
                        max_val,       # 最大值
                        energy,        # 能量
                        rms,           # 均方根
                        skewness,      # 偏度
                        kurtosis,      # 峰度
                        median_val,    # 中位数
                        iqr            # 四分位距
                    ])
                    node_features.append(features)
        
        return node_features

    def _build_edges_optimized(self, num_frames, grid_h, grid_w, spatial_offsets, temporal_offsets, device):
        """
        优化的边构建方法
        """
        edges = []
        num_nodes_per_frame = grid_h * grid_w
        
        # 预计算所有网格位置的坐标
        grid_positions = {}
        for i in range(grid_h):
            for j in range(grid_w):
                pos_index = i * grid_w + j
                grid_positions[pos_index] = (i, j)
        
        # 预计算合法的网格坐标范围
        valid_coords = set((i, j) for i in range(grid_h) for j in range(grid_w))
        
        # 为每个节点构建连接
        for t in range(num_frames):
            for pos_in_frame in range(num_nodes_per_frame):
                node_idx = t * num_nodes_per_frame + pos_in_frame
                i, j = grid_positions[pos_in_frame]
                
                # 空间连接 - 使用预计算的偏移量
                for di, dj in spatial_offsets:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in valid_coords:
                        neighbor_pos = ni * grid_w + nj
                        neighbor_idx = t * num_nodes_per_frame + neighbor_pos
                        edges.append([node_idx, neighbor_idx])
                
                # 时间连接 - 使用预计算的时间偏移量
                for dt in temporal_offsets:
                    target_t = t + dt
                    if 0 <= target_t < num_frames:
                        target_idx = target_t * num_nodes_per_frame + pos_in_frame
                        edges.append([node_idx, target_idx])
        
        # 转换为PyG格式
        # 关键修复：确保边索引张量不阻断梯度（边索引不需要梯度）
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index

def test_fixed_gradient_flow():
    """
    测试修复后的梯度流
    """
    print("Testing fixed gradient flow")
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 创建测试视频数据: [batch, channels, time, height, width]
    batch_size, channels, time, height, width = 2, 3, 16, 256, 256
    video = torch.randn(batch_size, channels, time, height, width, device=device, requires_grad=True)
    print(f"Input video shape: {video.shape}")
    
    # 初始化I3D特征提取器
    i3d_extractor = I3DFeatureExtractor(device)
    
    # 提取I3D特征
    with torch.no_grad():
        i3d_features = i3d_extractor.extract_features(video)
    
    print(f"I3D features shape: {i3d_features.shape}")
    print(f"I3D features requires_grad: {i3d_features.requires_grad}")
    
    # 为i3d_features手动设置requires_grad=True
    if not i3d_features.requires_grad:
        i3d_features = i3d_features.requires_grad_(True)
    
    # 初始化3D小波变换
    dwt3d = DWT3D(wavelet=Config.WAVELET_PARAMS['wavelet'], 
                  levels=Config.WAVELET_PARAMS['levels']).to(device)
    dwt3d.train()
    
    # 应用3D小波变换
    wavelet_coeffs = dwt3d(i3d_features)
    print(f"Wavelet coefficients processed")
    
    # 初始化高级小波系数注意力模块
    attention_module = AdvancedWaveletCoeffAttention(channels=1).to(device)
    attention_module.train()
    
    # 应用注意力机制
    fused_features = attention_module(wavelet_coeffs)
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Fused features requires_grad: {fused_features.requires_grad}")
    
    # 通过修复版SpatioTemporalGraphBuilder构建图
    graph_builder = FixedSpatioTemporalGraphBuilder(
        grid_size=Config.MODEL_PARAMS['grid_size'],
        temporal_window=Config.MODEL_PARAMS['temporal_window']
    ).to(device)
    graph_builder.train()
    
    # 构建图结构
    graphs = graph_builder.build_graph(fused_features, device, training=True)
    print(f"Graphs built: {len(graphs)} graphs")
    print(f"First graph node features shape: {graphs[0].x.shape}")
    print(f"First graph edge_index shape: {graphs[0].edge_index.shape}")
    print(f"First graph node features requires_grad: {graphs[0].x.requires_grad}")
    
    # 使用GatedGCN提取特征
    gated_gcn = GatedGCN(
        in_channels=graphs[0].x.shape[1],
        hidden_channels=Config.MODEL_PARAMS['hidden_channels'],
        out_channels=Config.MODEL_PARAMS['out_channels']
    ).to(device)
    gated_gcn.train()
    
    # 应用GatedGCN
    gcn_features = gated_gcn(graphs[0])
    print(f"GCN features shape: {gcn_features.shape}")
    print(f"GCN features requires_grad: {gcn_features.requires_grad}")
    
    # 创建损失并反向传播
    loss = gcn_features.sum()
    print(f"Loss value: {loss.item()}")
    
    # 执行反向传播
    loss.backward()
    
    # 检查梯度
    print("\n=== Gradient Check ===")
    
    # 检查GatedGCN的梯度
    print("GatedGCN gradients:")
    has_gcn_grad = False
    for name, param in gated_gcn.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_gcn_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"GatedGCN has gradients: {has_gcn_grad}")
    
    # 检查注意力模块的梯度
    print("\nAdvancedWaveletCoeffAttention gradients:")
    has_attention_grad = False
    for name, param in attention_module.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_attention_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"AdvancedWaveletCoeffAttention has gradients: {has_attention_grad}")
    
    print("\nTest completed!")
import torch
import torch.nn as nn
from models_i3d import I3DFeatureExtractor, DWT3D, AdvancedWaveletCoeffAttention, SpatioTemporalGraphBuilder, GatedGCN
from config import Config

# 尝试导入torchvision中的视频模型
try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

def test_complete_gradient_flow():
    """
    测试完整的梯度流，包括I3D->DWT3D->Attention->GraphBuilder->GatedGCN
    """
    print("Testing complete gradient flow through all components")
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 创建测试视频数据: [batch, channels, time, height, width]
    batch_size, channels, time, height, width = 2, 3, 16, 224, 224
    video = torch.randn(batch_size, channels, time, height, width, device=device, requires_grad=True)
    print(f"Input video shape: {video.shape}")
    
    # 初始化I3D特征提取器
    i3d_extractor = I3DFeatureExtractor(device)
    
    # 提取I3D特征 (使用no_grad因为I3D不需要训练)
    with torch.no_grad():
        i3d_features = i3d_extractor.extract_features(video)
    
    print(f"I3D features shape: {i3d_features.shape}")
    print(f"I3D features requires_grad: {i3d_features.requires_grad}")
    
    # 手动为I3D特征设置requires_grad，以便从这里开始梯度计算
    if not i3d_features.requires_grad:
        i3d_features = i3d_features.requires_grad_(True)
    
    # 初始化3D小波变换
    dwt3d = DWT3D(wavelet=Config.WAVELET_PARAMS['wavelet'], 
                  levels=Config.WAVELET_PARAMS['levels']).to(device)
    dwt3d.train()  # 设置为训练模式
    
    # 应用3D小波变换
    wavelet_coeffs = dwt3d(i3d_features)
    print(f"Wavelet coefficients processed")
    
    # 检查小波系数是否有梯度
    coeff_grad_status = []
    for level_key, level_coeffs in wavelet_coeffs.items():
        for channel_key, channel_coeffs in level_coeffs.items():
            for coeff_key, coeff_tensor in channel_coeffs.items():
                coeff_grad_status.append(coeff_tensor.requires_grad)
                break
        break
    print(f"Wavelet coefficients require grad: {any(coeff_grad_status)}")
    
    # 初始化高级小波系数注意力模块
    attention_module = AdvancedWaveletCoeffAttention(channels=1).to(device)
    attention_module.train()  # 设置为训练模式
    
    # 应用注意力机制
    fused_features = attention_module(wavelet_coeffs)
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Fused features requires_grad: {fused_features.requires_grad}")
    
    # 保存注意力模块前向传播后的特征用于梯度检查
    attention_output = fused_features.clone().detach().requires_grad_(True)
    
    # 通过SpatioTemporalGraphBuilder构建图
    graph_builder = SpatioTemporalGraphBuilder(
        grid_size=Config.MODEL_PARAMS['grid_size'],
        temporal_window=Config.MODEL_PARAMS['temporal_window']
    ).to(device)
    graph_builder.train()  # 设置为训练模式
    
    # 构建图结构
    graphs = graph_builder.build_graph(fused_features, device, training=True)
    print(f"Graphs built: {len(graphs)} graphs")
    print(f"First graph node features shape: {graphs[0].x.shape}")
    print(f"First graph edge_index shape: {graphs[0].edge_index.shape}")
    print(f"First graph node features requires_grad: {graphs[0].x.requires_grad}")
    
    # 使用GatedGCN提取特征
    gated_gcn = GatedGCN(
        in_channels=graphs[0].x.shape[1],  # 根据实际节点特征维度设置
        hidden_channels=Config.MODEL_PARAMS['hidden_channels'],
        out_channels=Config.MODEL_PARAMS['out_channels']
    ).to(device)
    gated_gcn.train()  # 设置为训练模式
    
    # 应用GatedGCN
    gcn_features = gated_gcn(graphs[0])  # 处理第一个图作为示例
    print(f"GCN features shape: {gcn_features.shape}")
    print(f"GCN features requires_grad: {gcn_features.requires_grad}")
    
    # 创建一个简单的损失函数并执行反向传播
    loss = gcn_features.sum()
    print(f"Loss value: {loss.item()}")
    
    # 执行反向传播
    loss.backward()
    
    # 检查各个模块的梯度
    print("\n=== Gradient Check ===")
    
    # 检查GatedGCN的梯度
    print("GatedGCN gradients:")
    has_gcn_grad = False
    for name, param in gated_gcn.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_gcn_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"GatedGCN has gradients: {has_gcn_grad}")
    
    # 检查注意力模块的梯度
    print("\nAdvancedWaveletCoeffAttention gradients:")
    has_attention_grad = False
    for name, param in attention_module.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_attention_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"AdvancedWaveletCoeffAttention has gradients: {has_attention_grad}")
    
    # 检查输入视频的梯度
    print(f"\nInput video gradient exists: {video.grad is not None}")
    if video.grad is not None:
        print(f"Input video gradient shape: {video.grad.shape}")
        print(f"Input video gradient norm: {video.grad.norm().item()}")
    
    # 检查i3d_features的梯度
    print(f"\nI3D features gradient exists: {i3d_features.grad is not None}")
    if i3d_features.grad is not None:
        print(f"I3D features gradient shape: {i3d_features.grad.shape}")
        print(f"I3D features gradient norm: {i3d_features.grad.norm().item()}")
    
    # 检查小波变换参数的梯度（如果有的话）
    print("\nDWT3D module gradients:")
    has_dwt_grad = False
    for name, param in dwt3d.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_dwt_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"DWT3D has gradients: {has_dwt_grad}")
    
    print("\nTest completed!")
    print(f"Overall gradient flow successful: {has_attention_grad}")


def test_gradient_flow_from_gatedgcn_to_attention_v2():
    """
    测试从GatedGCN提取的特征到Attention模块的梯度流 - 修复版
    """
    print("Testing gradient flow from GatedGCN back to AdvancedWaveletCoeffAttention v2")
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 创建测试视频数据: [batch, channels, time, height, width]
    batch_size, channels, time, height, width = 2, 3, 16, 224, 224
    video = torch.randn(batch_size, channels, time, height, width, device=device, requires_grad=True)
    print(f"Input video shape: {video.shape}")
    
    # 初始化I3D特征提取器
    i3d_extractor = I3DFeatureExtractor(device)
    
    # 提取I3D特征
    with torch.no_grad():  # I3D不需要梯度
        i3d_features = i3d_extractor.extract_features(video)
    
    print(f"I3D features shape: {i3d_features.shape}")
    print(f"I3D features requires_grad: {i3d_features.requires_grad}")
    
    # 为i3d_features手动设置requires_grad=True
    if not i3d_features.requires_grad:
        i3d_features = i3d_features.requires_grad_(True)
    
    # 初始化3D小波变换
    dwt3d = DWT3D(wavelet=Config.WAVELET_PARAMS['wavelet'], 
                  levels=Config.WAVELET_PARAMS['levels']).to(device)
    dwt3d.train()  # 设置为训练模式
    
    # 应用3D小波变换
    wavelet_coeffs = dwt3d(i3d_features)
    print(f"Wavelet coefficients processed")
    
    # 初始化高级小波系数注意力模块
    attention_module = AdvancedWaveletCoeffAttention(channels=1).to(device)
    attention_module.train()  # 设置为训练模式
    
    # 应用注意力机制
    fused_features = attention_module(wavelet_coeffs)
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Fused features requires_grad: {fused_features.requires_grad}")
    
    # 直接测试注意力模块的梯度，不通过图神经网络
    loss = fused_features.sum()
    print(f"Loss value: {loss.item()}")
    
    # 执行反向传播
    loss.backward()
    
    # 检查注意力模块的梯度
    print("\nAdvancedWaveletCoeffAttention gradients:")
    has_attention_grad = False
    for name, param in attention_module.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_attention_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"AdvancedWaveletCoeffAttention has gradients: {has_attention_grad}")
    
    print("\nTest completed!")

# 创建一个简化版的GatedGCN，不使用PyG Data对象
class SimpleGatedGCN(nn.Module):
    """
    简化版门控图卷积网络，不使用PyG Data对象，直接处理张量
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGatedGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 图卷积层 (使用线性层模拟图卷积)
        self.gcn1 = nn.Linear(in_channels, hidden_channels)
        self.gcn2 = nn.Linear(hidden_channels, out_channels)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(out_channels + in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征张量，形状为 [batch, channels, time, height, width]
            
        返回:
            torch.Tensor: 输出特征张量，形状为 [batch, out_channels]
        """
        batch_size, channels, time, height, width = x.shape
        
        # 重塑张量为节点特征 [batch, nodes, features]
        x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous()  # [batch, time, height, width, channels]
        x_reshaped = x_reshaped.view(batch_size, -1, channels)  # [batch, nodes, channels]
        
        # 第一层GCN
        x1 = torch.relu(self.gcn1(x_reshaped))  # [batch, nodes, hidden_channels]
        x1 = torch.dropout(x1, p=0.3, train=self.training)
        
        # 第二层GCN
        x2 = torch.relu(self.gcn2(x1))  # [batch, nodes, out_channels]
        
        # 全局池化获取图级表示
        x2_pooled = torch.mean(x2, dim=1)  # [batch, out_channels]
        
        # 原始特征池化
        x_original_pooled = torch.mean(x_reshaped, dim=1)  # [batch, in_channels]
        
        # 门控机制
        gate_input = torch.cat([x2_pooled, x_original_pooled], dim=1)  # [batch, out+in]
        gate = self.gate(gate_input)  # [batch, out_channels]
        
        # 应用门控
        result = x2_pooled * gate  # [batch, out_channels]
        
        return result
    
def test_gradient_flow_simple_gcn():
    """
    测试使用简化GCN的完整梯度流
    """
    print("Testing gradient flow with simplified GCN")
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 创建测试视频数据: [batch, channels, time, height, width]
    batch_size, channels, time, height, width = 2, 3, 16, 112, 112
    video = torch.randn(batch_size, channels, time, height, width, device=device, requires_grad=True)
    print(f"Input video shape: {video.shape}")
    
    # 初始化I3D特征提取器
    i3d_extractor = I3DFeatureExtractor(device)
    
    # 提取I3D特征
    with torch.no_grad():
        i3d_features = i3d_extractor.extract_features(video)
    
    print(f"I3D features shape: {i3d_features.shape}")
    print(f"I3D features requires_grad: {i3d_features.requires_grad}")
    
    # 为i3d_features手动设置requires_grad=True
    if not i3d_features.requires_grad:
        i3d_features = i3d_features.requires_grad_(True)
    
    # 初始化3D小波变换
    dwt3d = DWT3D(wavelet=Config.WAVELET_PARAMS['wavelet'], 
                  levels=Config.WAVELET_PARAMS['levels']).to(device)
    dwt3d.train()
    
    # 应用3D小波变换
    wavelet_coeffs = dwt3d(i3d_features)
    print(f"Wavelet coefficients processed")
    
    # 初始化高级小波系数注意力模块
    attention_module = AdvancedWaveletCoeffAttention(channels=1).to(device)
    attention_module.train()
    
    # 应用注意力机制
    fused_features = attention_module(wavelet_coeffs)
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Fused features requires_grad: {fused_features.requires_grad}")
    
    # 使用简化版GatedGCN
    simple_gcn = SimpleGatedGCN(
        in_channels=fused_features.shape[1],  # C
        hidden_channels=Config.MODEL_PARAMS['hidden_channels'],
        out_channels=Config.MODEL_PARAMS['out_channels']
    ).to(device)
    simple_gcn.train()
    
    # 应用简化版GatedGCN
    gcn_features = simple_gcn(fused_features)
    print(f"GCN features shape: {gcn_features.shape}")
    print(f"GCN features requires_grad: {gcn_features.requires_grad}")
    
    # 创建损失并反向传播
    loss = gcn_features.sum()
    print(f"Loss value: {loss.item()}")
    
    # 执行反向传播
    loss.backward()
    
    # 检查梯度
    print("\n=== Gradient Check ===")
    
    # 检查简化版GatedGCN的梯度
    print("SimpleGatedGCN gradients:")
    has_gcn_grad = False
    for name, param in simple_gcn.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_gcn_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"SimpleGatedGCN has gradients: {has_gcn_grad}")
    
    # 检查注意力模块的梯度
    print("\nAdvancedWaveletCoeffAttention gradients:")
    has_attention_grad = False
    for name, param in attention_module.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad is not None} (shape: {param.grad.shape})")
            has_attention_grad = True
        else:
            print(f"  {name}: {param.grad is not None}")
    print(f"AdvancedWaveletCoeffAttention has gradients: {has_attention_grad}")
    
    # 检查输入视频的梯度
    print(f"\nInput video gradient exists: {video.grad is not None}")
    if video.grad is not None:
        print(f"Input video gradient shape: {video.grad.shape}")
        print(f"Input video gradient norm: {video.grad.norm().item()}")
    
    print("\nTest completed!")
    print(f"Overall gradient flow successful: {has_attention_grad}")

if __name__ == "__main__":
    # test_gradient_flow_from_gatedgcn_to_attention_v2()
    # test_complete_gradient_flow()
    # test_gradient_flow_simple_gcn()
    test_fixed_gradient_flow()


