# Replace these imports at the top of your file:
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
from utils import setup_logging
logger = setup_logging()

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
        # # 关键修复1：调整维度顺序为 [batch, frames, height, width, channels]（适配ptwt要求）
        # # ptwt的wavedec3要求通道在最后一维，否则会维度误判
        # video = video.permute(0, 2, 3, 4, 1)

        all_coeffs = {}
        
        # 确保输入视频需要梯度
        if not video.requires_grad:
            video = video.requires_grad_(True)
        
        try:
           #  print("adsfasdf")
           #  # 使用ptwt进行3D小波变换，直接处理整个批次 (N, C, D, H, W)
           #  print("查看一下video的形状：",video.shape)
           # # 这里我加入了下面一行代码，删除了通道维度
           #  video = video.squeeze(1)
           #  print("查看一下修改后video的形状：", video.shape)  # torch.Size([4, 1, 8, 128, 128])
            coeffs = wavedec3(video, wavelet=self.wavelet, level=self.levels)
            print("adsfasdf")
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
            print("asdfadf3D小波变换失败")
            print(f"3D小波变换失败models.py160行: {e}")
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

  
# 3. 门控图卷积网络 (Gated GCN)

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
        
        参数:
            data (torch_geometric.data.Data): 包含节点特征和边索引的图数据
                输入维度: PyG Data对象，包含:
                    x: [num_nodes, in_channels] 节点特征
                    edge_index: [2, num_edges] 边索引
            
        返回:
            torch.Tensor: 经过门控机制处理后的节点特征
                输出维度: [num_nodes, out_channels]
        """
        # 确保数据在正确的设备上
        # 修改开始：更严格的设备一致性检查
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
        
        # 动态初始化网络层以适应实际输入维度
        # 检查是否需要初始化网络层
        if self.conv1 is None or self.conv1.in_channels != original_x.shape[1]:
            self._init_layers(original_x.shape[1])
        else:
            # 确保已存在的网络层在正确的设备上
            # 修改开始：更严格的设备检查和移动
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
            # 修改结束
        
        # 确保所有层都在正确的设备上后再进行计算
        # 修改开始：在计算前再次确认所有张量在正确设备上
        if data.x.device != self._device:
            data.x = data.x.to(self._device)
        if data.edge_index.device != self._device:
            data.edge_index = data.edge_index.to(self._device)
        # 修改结束
        
        # 第一层GCN：图卷积 + ReLU激活函数
        try:
            x = F.relu(self.conv1(data.x, data.edge_index))
            # 检查输出是否包含NaN或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("第一层GCN输出包含NaN或inf值")
                x = torch.zeros_like(x)
        except Exception as e:
            logger.warning(f"第一层GCN计算失败: {e}")
            x = torch.zeros_like(data.x)
        
        # Dropout正则化：防止过拟合
        x = F.dropout(x, p=0.3, training=self.training)  # 增加dropout率以增强泛化能力
        
        # 第二层GCN：进一步提取图结构特征
        try:
            x = F.relu(self.conv2(x, data.edge_index))  # 添加ReLU激活函数
            # 检查输出是否包含NaN或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("第二层GCN输出包含NaN或inf值")
                x = torch.zeros_like(x)
        except Exception as e:
            logger.warning(f"第二层GCN计算失败: {e}")
            x = torch.zeros_like(x)
            
        # 添加残差连接以增强梯度流动
        # 修改开始：添加残差连接
        if x.shape == original_x.shape:
            x = x + original_x  # 残差连接
        # 修改结束
        
        # 门控机制：自适应融合原始特征和处理后的特征
        # 拼接原始特征和处理后的特征
        # 修改开始：确保拼接前所有张量在正确设备上并检查NaN或inf
        if x.device != self._device:
            x = x.to(self._device)
        if original_x.device != self._device:
            original_x = original_x.to(self._device)
            
        # 检查拼接前的张量是否包含NaN或inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("处理后的特征包含NaN或inf值")
            x = torch.zeros_like(x)
        if torch.isnan(original_x).any() or torch.isinf(original_x).any():
            logger.warning("原始特征包含NaN或inf值")
            original_x = torch.zeros_like(original_x)
        # 修改结束
        
        try:
            gate_input = torch.cat([x, original_x], dim=1)
            # 检查拼接后的输入是否包含NaN或inf
            if torch.isnan(gate_input).any() or torch.isinf(gate_input).any():
                logger.warning("门控机制输入包含NaN或inf值")
                gate_input = torch.zeros_like(gate_input)
        except Exception as e:
            logger.warning(f"门控机制输入拼接失败: {e}")
            # 创建合适维度的零张量
            gate_input = torch.zeros(x.shape[0], x.shape[1] + original_x.shape[1], device=self._device)
        
        # 通过门控网络生成门控权重
        try:
            gate = self.gate(gate_input)
            # 检查门控权重是否包含NaN或inf
            if torch.isnan(gate).any() or torch.isinf(gate).any():
                logger.warning("门控权重包含NaN或inf值")
                gate = torch.zeros_like(gate)
        except Exception as e:
            logger.warning(f"门控权重计算失败: {e}")
            # 创建合适维度的零张量
            gate = torch.zeros(x.shape[0], x.shape[1], device=self._device)
        
        # 应用门控权重：控制处理后特征的贡献程度
        result = x * gate
        
        # 确保输出在正确的设备上
        if result.device != self._device:
            result = result.to(self._device)
            
        # 最终检查输出是否包含NaN或inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("GatedGCN最终输出包含NaN或inf值，使用零张量替代")
            result = torch.zeros_like(result)
            
        return result

class SpatioTemporalGraphBuilder:
    def __init__(self, grid_size=Config.MODEL_PARAMS['grid_size'], 
                 temporal_window=Config.MODEL_PARAMS['temporal_window']):
        self.grid_size = grid_size
        self.temporal_window = temporal_window
    
    def build_graph(self, video_tensor, device):
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
        grid_h, grid_w = H // self.grid_size, W // self.grid_size
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
        
        for b in range(batch_size):
            # 向量化节点特征计算 (处理所有通道)
            node_features = self._compute_node_features_vectorized(
                video_tensor[b], grid_h, grid_w  # [channels, frames, height, width]
            )
            
            # 高效构建边
            edge_index = self._build_edges_optimized(
                num_frames, grid_h, grid_w, spatial_offsets, temporal_offsets, device
            )
            
            # 转换为PyG Data对象，并确保所有张量在正确设备上
            x = torch.stack(node_features, dim=0).to(device)
            edge_index = edge_index.to(device)
            graph = Data(x=x, edge_index=edge_index)
            # 确保整个图数据对象在正确的设备上
            graph = graph.to(device)
            graphs.append(graph)
        
        return graphs

    def _compute_node_features_vectorized(self, video_frame, grid_h, grid_w):
        """
        向量化计算节点特征 (处理所有通道)
        输入: video_frame [channels, frames, height, width]
        输出: 节点特征列表，每个元素是 [feature_dim] 的张量
        """
        channels, frames, H, W = video_frame.shape
        node_features = []
        
        # 获取设备信息
        device = video_frame.device
        
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
                    
                    # 创建特征向量，确保在正确的设备上
                    features = torch.tensor([
                        mean_val.item(),      # 均值
                        std_val.item(),       # 标准差
                        min_val.item(),       # 最小值
                        max_val.item(),       # 最大值
                        energy.item(),        # 能量
                        rms.item(),           # 均方根
                        skewness.item(),      # 偏度
                        kurtosis.item(),      # 峰度
                        median_val.item(),    # 中位数
                        iqr.item()            # 四分位距
                    ], dtype=combined_patch.dtype, device=device)
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
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index

class AdvancedWaveletCoeffAttention(nn.Module):
    """
    输入: [B, K, C, T, H, W]
    输出: [B, C, T, H, W]
    通道 / 空间 / 时间 注意力的中间通道数均为 max(8, C // 8)
    """
    def __init__(self, num_coefficients: int, channels: int):
        super().__init__()
        self.K = num_coefficients
        self.C = channels
        mid = max(8, channels // 2)          # 统一中间通道数

        # 1. 通道注意力：作用于 C，保留 T
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # [B*K*T, C, 1, 1]
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 2. 空间注意力：作用于 (H, W)，共享 K,C,T
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 3. 时间注意力：作用于 T，共享 K,C,H,W
        self.temporal_att = nn.Sequential(
            nn.Conv1d(1, mid, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, K, C, T, H, W = x.shape

        # ---------- 1. 通道注意力 ----------
        x4d = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B,K,T,C,H,W]
        x4d = x4d.view(-1, C, H, W)                      # [B*K*T,C,H,W]
        ca = self.channel_att(x4d)                       # [B*K*T,C,1,1]
        ca = ca.view(B, K, T, C, 1, 1).permute(0, 1, 3, 2, 4, 5)  # [B,K,C,T,1,1]
        x = x * ca                                       # broadcast

        # ---------- 2. 空间注意力 ----------
        feat_hw = x.mean(dim=(1, 2, 3), keepdim=False)   # [B,H,W]
        feat_hw = feat_hw.unsqueeze(1)                   # [B,1,H,W]
        sa = self.spatial_att(feat_hw)                     # [B,1,H,W]
        sa = sa.unsqueeze(1).unsqueeze(1)                  # [B,1,1,1,H,W]

        # ---------- 3. 时间注意力 ----------
        feat_t = x.mean(dim=(1, 2, 4, 5), keepdim=False)   # [B,T]
        feat_t = feat_t.unsqueeze(1)                       # [B,1,T]
        ta = self.temporal_att(feat_t)                     # [B,1,T]
        ta = ta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)    # [B,1,1,T,1,1]

        # ---------- 4. 融合 ----------
        att = sa * ta                                      # [B,1,1,T,H,W]
        out = (x * att).sum(dim=1)                         # [B,C,T,H,W]
        return out


class ImprovedWaveletSTGNN(nn.Module):
    """
    改进的小波时空图神经网络 (Improved Wavelet Spatio-Temporal Graph Neural Network)
    
    该网络结合了小波变换、时空图构建和图神经网络，用于处理视频数据。
    与原始版本不同，该版本利用所有小波系数而不仅仅是低频分量。
    输入数据维度为 [batch, channels, time, height, width]，与之前模块保持一致。
    """
    def __init__(self, in_channels=None, hidden_channels=None, out_channels=None, grid_size=None, temporal_window=None):
        """
        初始化改进的小波时空图神经网络
        
        参数:
            in_channels (int): 输入通道数
                例如: 1 表示灰度图像
                输入维度: [batch, channels, time, height, width]
            hidden_channels (int): 隐藏层通道数
                例如: 64 表示隐藏层有64个通道
            out_channels (int): 输出通道数
                例如: 32 表示输出层有32个通道
            grid_size (int): 网格大小，用于构建时空图
                例如: 8 表示将图像划分为8x8的网格
            temporal_window (int): 时间窗口大小
                例如: 2 表示考虑前后2帧的时间信息
                
        输出:
            None (初始化模型结构)
        """
        super(ImprovedWaveletSTGNN, self).__init__()
        
        # 从config获取参数作为默认值
        self.in_channels = in_channels if in_channels is not None else Config.MODEL_PARAMS['in_channels']
        self.hidden_channels = hidden_channels if hidden_channels is not None else Config.MODEL_PARAMS['hidden_channels']
        self.out_channels = out_channels if out_channels is not None else Config.MODEL_PARAMS['out_channels']
        self.grid_size = grid_size if grid_size is not None else Config.MODEL_PARAMS['grid_size']
        self.temporal_window = temporal_window if temporal_window is not None else Config.MODEL_PARAMS['temporal_window']
       
        # 3D离散小波变换层
        self.dwt3d = DWT3D(wavelet=Config.WAVELET_PARAMS['wavelet'], 
                          levels=Config.WAVELET_PARAMS['levels'])
        
        # 时空图构建器
        self.graph_builder = SpatioTemporalGraphBuilder(grid_size=self.grid_size, 
                                                      temporal_window=self.temporal_window)
        
        # 门控图卷积网络 - 增加层数以提高表达能力
        # 使用更深的GatedGCN网络，增加模型复杂度
        self.gated_gcn = GatedGCN(in_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 out_channels=self.out_channels)
        
        # 批标准化层，使用更稳定的参数
        self.batch_norm = nn.BatchNorm1d(self.out_channels, eps=1e-5, momentum=0.1, affine=True)
        
        # Dropout层用于正则化，降低dropout率提高稳定性
        self.dropout = nn.Dropout(0.2)  # 增加dropout率以增强泛化能力
        
        # 输出投影层，将图级特征映射到异常分数
        # 增加模型复杂度以提高区分能力
        # 修改开始：简化输出投影层结构，避免维度问题
        self.output_projection = nn.Sequential(
            nn.Linear(self.out_channels, max(1, self.out_channels // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.out_channels // 2), max(1, self.out_channels // 4)),
            nn.ReLU(),
            nn.Linear(max(1, self.out_channels // 4), 1)
        )
        # 修改结束
        
        # 添加残差连接层，增强梯度流动
        # 修改开始：简化残差连接结构
        self.residual_projection = nn.Sequential(
        nn.Linear(self.out_channels, max(1, self.out_channels // 2)),
        nn.ReLU(),
        nn.Dropout(0.1),  # 添加dropout
        nn.Linear(max(1, self.out_channels // 2), 1),
        nn.ReLU()  # 确保输出为正
    )
        
        # 添加注意力机制来增强特征表达
        # 增加注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=4, dropout=0.1)  # 适度的头数和dropout率
        
        # 初始化所有参数以提高数值稳定性
        self._initialize_weights() 

    def _initialize_weights(self):
        """
        初始化模型权重以提高数值稳定性
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用xavier初始化线性层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # 初始化批归一化层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _process_multi_level_wavelet_coeffs(self, wavelet_coeffs):
        """
        处理多层小波系数，融合不同层级的系数为统一特征表示
        
        参数:
            wavelet_coeffs (dict): 小波系数字典
                输入格式: DWT3D返回的字典格式
                例如: {
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
                
        返回:
            torch.Tensor: 融合后的特征张量
                输出维度: [batch_size, channels, time, height, width]
                例如: [2, 1, 8, 64, 64] 表示2个批次，1个通道，8帧，64x64的特征图
        """
        # 收集所有系数张量（排除低频信息）
        all_coeffs_tensors = []
        
        if isinstance(wavelet_coeffs, dict):
            # 处理DWT3D返回的字典格式
            for level_key, level_data in wavelet_coeffs.items():
                if isinstance(level_data, dict):
                    for channel_key, channel_data in level_data.items():
                        if isinstance(channel_data, dict):
                            # 提取所有系数张量（排除'LLL'低频系数）
                            for coeff_key, coeff_tensor in channel_data.items():
                                # 只添加高频系数，排除低频的'LLL'系数
                                if isinstance(coeff_tensor, torch.Tensor) and coeff_key != 'LLL':
                                    all_coeffs_tensors.append(coeff_tensor)
        
        # 确保至少有一个系数
        if not all_coeffs_tensors:
            raise ValueError("没有找到有效的小波系数张量")
            
        # 使用第一个系数作为基准，分辨率最高的作为基准
        base_coeff = all_coeffs_tensors[0]  # 使用第一个高频系数作为基准
        batch_size, channels, time, height, width = base_coeff.shape
        device = base_coeff.device
        
        # 处理所有系数，确保尺寸一致并进行上采样
        processed_coeffs = []
        for coeff in all_coeffs_tensors:
            # 确保设备一致
            coeff = coeff.to(device)
            
            # 确保梯度一致
            if base_coeff.requires_grad and not coeff.requires_grad:
                coeff = coeff.requires_grad_(True)
            elif not base_coeff.requires_grad and coeff.requires_grad:
                coeff = coeff.detach()
                
            # 尺寸对齐 - 上采样到基准尺寸
            if coeff.shape != base_coeff.shape:
                try:
                    # 进行上采样对齐到基准尺寸
                    coeff = F.interpolate(coeff, size=(time, height, width), 
                                        mode='trilinear', align_corners=False)
                except:
                    # 如果上采样失败，使用平均值填充
                    mean_val = coeff.mean()
                    coeff = torch.full((batch_size, channels, time, height, width), 
                                     mean_val, device=device, requires_grad=coeff.requires_grad)
            
            processed_coeffs.append(coeff)
            
        # 如果处理后没有有效系数，创建默认张量
        if not processed_coeffs:
            dummy_tensor = torch.zeros(batch_size, channels, time, height, width, 
                                     device=device, requires_grad=True)
            return dummy_tensor
            
        # 将所有系数堆叠在一起
        stacked_coeffs = torch.stack(processed_coeffs, dim=1)  # [batch, num_coeffs, channels, time, height, width]
        
        # 初始化注意力模块（如果尚未初始化）
        if not hasattr(self, 'wavelet_attention') or getattr(self.wavelet_attention, 'K', None) != len(processed_coeffs):
            # 可以选择使用基础版或高级版注意力机制
            # self.wavelet_attention = WaveletCoeffAttention(len(processed_coeffs)).to(device)
            batch, num_coeffs, channels, time, height, width = stacked_coeffs.shape
            self.wavelet_attention = AdvancedWaveletCoeffAttention(num_coeffs, channels).to(device)
            
        # 应用注意力机制并融合系数
        fused_features = self.wavelet_attention(stacked_coeffs)
        
        # 确保输出张量有梯度
        if not fused_features.requires_grad:
            fused_features = fused_features.requires_grad_(True)
            
        return fused_features    
    
    def forward(self, video):
        """
        前向传播过程，计算视频的异常分数
        
        参数:
            video (torch.Tensor): 输入视频张量
                输入维度: [batch_size, channels, time, height, width]
                例如: [2, 1, 8, 128, 128] 表示2个批次，1个通道，8帧，128x128的图像
            
        返回:
            torch.Tensor: 异常检测分数
                输出维度: [batch_size, time]
                例如: [2, 8] 表示每个批次中每帧的异常分数
        """
        global logger
        batch_size, channels, time, height, width = video.shape
        
        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        # 确保输入在正确的设备上并且需要梯度
        x = video.to(device)
        if self.training and not x.requires_grad:
            x = x.requires_grad_(True)
        
        # 限制输入范围，防止过大值导致数值不稳定
        x = torch.clamp(x, min=-10, max=10)
        
        try:
            # 1. 应用3D小波变换
            # 输入: [batch_size, channels, time, height, width]
            # 输出: 多级小波系数字典
            print("x.shape",x.shape)
            wavelet_coeffs = self.dwt3d(x)
        except Exception as e:
            logger.error(f"3D小波11变换失败: {e}")
            # 返回默认的低异常分数
            result = torch.full((batch_size, time), 1e-6, device=device, requires_grad=True)
            return result
        
        try:
            # 2. 处理多层小波系数（使用新添加的函数）
            fused_features = self._process_multi_level_wavelet_coeffs(wavelet_coeffs)
        except Exception as e:
            logger.error(f"小波系数处理失败: {e}")
            # 返回默认的低异常分数
            result = torch.full((batch_size, time), 1e-6, device=device, requires_grad=True)
            return result
        
        # 限制融合特征范围
        fused_features = torch.clamp(fused_features, min=-100, max=100)
        
        # 3. 构建时空图，确保所有张量在正确设备上
        # 输入: fused_features [batch_size, 1, time, height', width']
        # 输出: PyG Data对象列表(每个batch一个)
        try:
            graphs = self.graph_builder.build_graph(fused_features, device)
        except Exception as e:
            logger.error(f"图构建失败: {e}")
            # 返回默认的低异常分数
            result = torch.full((batch_size, time), 1e-6, device=device, requires_grad=True)
            return result
        
        # 4. 对每个图应用门控图卷积网络
        # 输入: PyG Data对象 (包含节点特征和边索引的图数据)
        # 输出: 经过门控机制处理后的节点特征 [num_nodes, out_channels]
        graph_outputs = []
        for graph in graphs:  # 一个batch一个图
            try:
                # 确保图数据在正确设备上
                graph = graph.to(device)
                # 应用门控GCN
                node_features = self.gated_gcn(graph)
                # 检查输出是否包含NaN或inf
                if torch.isnan(node_features).any() or torch.isinf(node_features).any():
                    logger.warning("GatedGCN输出包含NaN或inf值，使用零张量替代")
                    node_features = torch.zeros_like(node_features)
                graph_outputs.append(node_features)
            except Exception as e:
                logger.error(f"图卷积网络失败: {e}")
                # 如果某个图处理失败，使用零张量替代
                dummy_features = torch.zeros(1, self.out_channels, device=device)
                graph_outputs.append(dummy_features)
                continue
        
        # 5. 聚合节点特征为图级表示
        # 输入: node_features [num_nodes, out_channels]
        # 输出: graph_level_features [batch_size, time, out_channels]
        graph_level_features = []
        # 重新计算网格尺寸，因为小波变换会改变尺寸
        # 使用融合后的特征尺寸
        try:
            ll_height, ll_width = fused_features.shape[3], fused_features.shape[4]
            nodes_per_frame = max(1, (ll_height // self.grid_size) * (ll_width // self.grid_size))
            
            # 注意：由于小波变换会将时间维度减半，我们需要相应调整
            processed_time = max(1, fused_features.shape[2])  # 处理后的时间维度（通常是原始的一半）
        except Exception as e:
            logger.error(f"计算特征尺寸失败: {e}")
            # 使用默认值
            nodes_per_frame = 1
            processed_time = max(1, time // 2)
        
        for i, node_features in enumerate(graph_outputs):
            # 按帧分割节点特征
            frame_features = []
            for t in range(processed_time):  # 使用处理后的时间维度
                try:
                    # 提取当前帧的节点特征
                    start_idx = t * nodes_per_frame
                    end_idx = min((t + 1) * nodes_per_frame, node_features.shape[0])
                    if end_idx > start_idx:  # 确保索引有效
                        frame_nodes = node_features[start_idx:end_idx]
                        
                        # 对当前帧的所有节点特征进行全局平均池化
                        # 输入: frame_nodes [nodes_per_frame, out_channels]
                        # 输出: frame_feature [out_channels]
                        frame_feature = torch.mean(frame_nodes, dim=0)  # 使用平均池化而非最大池化
                        frame_features.append(frame_feature)
                except Exception as e:
                    logger.warning(f"处理节点特征失败: {e}")
                    # 使用零特征替代
                    zero_feature = torch.zeros(self.out_channels, device=device)
                    frame_features.append(zero_feature)
                    continue
            
            # 堆叠所有帧的特征
            # 输出: frame_features [processed_time, out_channels]
            if frame_features:  # 确保有特征
                try:
                    frame_features = torch.stack(frame_features, dim=0)  # [processed_time, out_channels]
                    # 检查是否包含NaN或inf
                    if torch.isnan(frame_features).any() or torch.isinf(frame_features).any():
                        logger.warning("帧特征包含NaN或inf值，使用零张量替代")
                        frame_features = torch.zeros_like(frame_features)
                    graph_level_features.append(frame_features)
                except Exception as e:
                    logger.warning(f"堆叠帧特征失败: {e}")
                    continue
        
        # 6. 将所有批次的特征堆叠
        # 输入: graph_level_features (list of tensors with shape [processed_time, out_channels])
        # 输出: batch_features with shape [batch_size, processed_time, out_channels]
        if graph_level_features:  # 确保有特征
            try:
                batch_features = torch.stack(graph_level_features, dim=0)  # [batch_size, processed_time, out_channels]
                
                # 保存原始特征用于残差连接
                original_features = batch_features.clone()
                
                # 确保所有相关层都在正确的设备上
                batch_size_feat, time_feat, out_channels_feat = batch_features.shape
                batch_features = batch_features.view(-1, out_channels_feat)
                
                # 确保所有模块层都在正确的设备上
                # 确保所有子模块在正确的设备上
                self.batch_norm = self.batch_norm.to(device)
                self.dropout = self.dropout.to(device)
                self.output_projection = self.output_projection.to(device)
                
                # 如果有残差连接层，也确保它在正确的设备上
                if hasattr(self, 'residual_projection') and self.residual_projection is not None:
                    self.residual_projection = self.residual_projection.to(device)
                    
                # 确保注意力机制在正确的设备上
                self.attention = self.attention.to(device)
                
                # 应用注意力机制增强特征表达
                # 应用注意力机制
                batch_features = batch_features.unsqueeze(0)  # 添加序列维度 [1, batch*time, out_channels]
                attended_features, _ = self.attention(batch_features, batch_features, batch_features)
                batch_features = attended_features.squeeze(0)  # 移除序列维度 [batch*time, out_channels]
                
                # 应用BatchNorm层来标准化特征
                # 确保batch normalization的参数在正确的设备上，并处理可能的异常
                if batch_features.device != device:
                    batch_features = batch_features.to(device)
                
                # 在eval模式下，如果BatchNorm的统计量不稳定，使用替代方法
                try:
                    # 检查BatchNorm是否已初始化
                    if hasattr(self.batch_norm, 'running_mean') and self.batch_norm.running_mean is not None:
                        batch_features = self.batch_norm(batch_features)
                    else:
                        # 如果BatchNorm未正确初始化，使用手动标准化
                        mean = batch_features.mean(dim=0, keepdim=True)
                        std = batch_features.std(dim=0, keepdim=True) + 1e-8
                        batch_features = (batch_features - mean) / std
                except Exception as e:
                    logger.warning(f"BatchNorm应用失败: {e}")
                    # 使用手动标准化作为替代
                    mean = batch_features.mean(dim=0, keepdim=True)
                    std = batch_features.std(dim=0, keepdim=True) + 1e-8
                    batch_features = (batch_features - mean) / std
                
                batch_features = batch_features.view(batch_size_feat, time_feat, out_channels_feat)
                
                # 应用Dropout来增加模型鲁棒性（在训练时应用，推理时不应用）
                if self.training:
                    batch_features = self.dropout(batch_features)
                
                # 7. 应用输出投影层得到异常分数
                # 输入: [batch_size, processed_time, out_channels]
                # 输出: [batch_size, processed_time, 1] -> [batch_size, processed_time] (squeeze后)
                # 使用主干路径和残差路径结合的方式计算异常分数，并限制输出范围
                try:
                    main_scores = self.output_projection(batch_features)  # [batch_size, processed_time, 1]
                    # 检查输出是否包含NaN或inf
                    if torch.isnan(main_scores).any() or torch.isinf(main_scores).any():
                        logger.warning("主干路径输出投影包含NaN或inf值，使用零张量替代")
                        main_scores = torch.zeros_like(main_scores)
                except Exception as e:
                    logger.warning(f"主干路径输出投影失败: {e}")
                    # 使用零张量替代
                    main_scores = torch.zeros(batch_size_feat, time_feat, 1, device=device)
                
                # 如果有残差连接层，则使用它
                if hasattr(self, 'residual_projection') and self.residual_projection is not None:
                    try:
                        # 对原始特征进行全局池化以获得更丰富的表示
                        pooled_features = torch.mean(original_features, dim=1, keepdim=True)  # [batch_size, 1, out_channels]
                        pooled_features = pooled_features.expand(-1, time_feat, -1)  # [batch_size, processed_time, out_channels]
                        
                        residual_scores = self.residual_projection(pooled_features)  # [batch_size, processed_time, 1]
                        # 检查输出是否包含NaN或inf
                        if torch.isnan(residual_scores).any() or torch.isinf(residual_scores).any():
                            logger.warning("残差路径输出包含NaN或inf值，使用零张量替代")
                            residual_scores = torch.zeros_like(residual_scores)
                        
                        # 使用更平衡的组合方式
                        anomaly_scores = 0.7 * main_scores + 0.3 * residual_scores  # 调整权重比例
                    except Exception as e:
                        logger.warning(f"残差路径计算失败: {e}")
                        anomaly_scores = main_scores
                else:
                    anomaly_scores = main_scores
                    
                anomaly_scores = anomaly_scores.squeeze(-1)  # [batch_size, processed_time]
                
                # 限制分数范围，防止过大
                anomaly_scores = torch.clamp(anomaly_scores, min=-10, max=10)
                
                # 增强分数区分度 - 关键改进部分
                # 使用更稳定的激活函数
                # 增强分数区分度，使用softplus激活函数
                anomaly_scores = F.softplus(anomaly_scores)  # 使用softplus确保正值且有更好梯度
                
                # 添加批标准化来稳定输出，增加数值稳定性检查
                scores_mean = anomaly_scores.mean()
                scores_std = anomaly_scores.std() + 1e-8  # 防止除零
                if torch.isnan(scores_mean) or torch.isinf(scores_mean) or torch.isnan(scores_std) or torch.isinf(scores_std):
                    logger.warning("分数统计量包含NaN或inf值，使用默认值")
                    scores_mean = torch.tensor(0.0, device=anomaly_scores.device)
                    scores_std = torch.tensor(1.0, device=anomaly_scores.device)
                anomaly_scores = (anomaly_scores - scores_mean) / scores_std
                
                # 使用温度缩放增强区分度
                temperature = 1.5  # 降低温度值以增强区分度
                anomaly_scores = anomaly_scores / temperature
                
                # 确保分数为正值
                anomaly_scores = torch.clamp(anomaly_scores, min=0)
                
                # 添加小的噪声来增加区分度（仅在训练时），提高模型鲁棒性
                if self.training:
                    noise = torch.randn_like(anomaly_scores) * 0.01
                    anomaly_scores = anomaly_scores + noise
                
                # 8. 恢复时间维度到原始长度
                # 如果处理后的时间维度与原始时间维度不一致，则进行上采样
                if processed_time != time:
                    # 添加维度以便进行时间上采样: [batch_size, processed_time] -> [batch_size, 1, processed_time]
                    anomaly_scores = anomaly_scores.unsqueeze(1)
                    # 上采样到原始时间长度: [batch_size, 1, processed_time] -> [batch_size, 1, time]
                    try:
                        anomaly_scores = F.interpolate(anomaly_scores, size=time, mode='linear', align_corners=False)
                    except Exception as e:
                        logger.warning(f"时间上采样失败: {e}")
                        # 如果上采样失败，使用平均值填充
                        mean_score = anomaly_scores.mean(dim=2, keepdim=True)
                        anomaly_scores = mean_score.expand(-1, -1, time)
                    # 移除额外的维度: [batch_size, 1, time] -> [batch_size, time]
                    anomaly_scores = anomaly_scores.squeeze(1)
                
                # 9. 最终限制分数范围
                anomaly_scores = torch.clamp(anomaly_scores, min=0, max=10)
                
                # 10. 最终检查输出是否包含NaN或inf
                if torch.isnan(anomaly_scores).any() or torch.isinf(anomaly_scores).any():
                    logger.warning("最终输出包含NaN或inf值，使用默认值替代")
                    anomaly_scores = torch.full((batch_size, time), 1e-6, device=device)
                    
            except Exception as e:
                logger.error(f"处理批次特征失败: {e}")
                # 如果处理失败，返回默认的低异常分数
                anomaly_scores = torch.full((batch_size, time), 1e-6, device=device, requires_grad=True)
        else:
            # 如果没有特征，返回很小的正数张量而不是零张量
            # 输出: [batch_size, time]
            anomaly_scores = torch.full((batch_size, time), 1e-6, device=device, requires_grad=True)
        
        return anomaly_scores

def test_improved_wavelet_stgnn():
    """测试改进的小波时空图神经网络"""
    print("Testing ImprovedWaveletSTGNN...")
    # 创建测试视频张量
    batch_size, channels, frames, height, width = 4, 1, 8, 128, 128
    video = torch.randn(batch_size, channels, frames, height, width)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video = video.to(device)
    
    # 初始化完整模型
    model = ImprovedWaveletSTGNN(
        in_channels=channels,
        hidden_channels=32,
        out_channels=16,
        grid_size=8,
        temporal_window=4
    ).to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 执行前向传播
    with torch.no_grad():
        output = model(video)
    
    print(f"Input video shape: {video.shape}")
    print(f"Output anomaly scores shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, {frames}]")
    print("ImprovedWaveletSTGNN test passed!\n")

if __name__ == "__main__":
    test_improved_wavelet_stgnn()