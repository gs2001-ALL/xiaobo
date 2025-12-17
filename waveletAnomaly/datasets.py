import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
from config import Config

class VideoAnomalyDataset(Dataset):
    """
    视频异常检测数据集类（支持视频分段）
    """
    
    def __init__(self, root_dir, label_file, file_type=Config.FILE_TYPE, 
                 transform=None, max_frames=Config.MAX_FRAMES, 
                 segment_length=Config.DATASET_PARAMS['segment_length'], 
                 overlap_ratio=Config.DATASET_PARAMS['overlap_ratio']):
        """
        初始化数据集
        
        参数:
            root_dir (str): 数据根目录路径
            label_file (str): 标签文件路径
            file_type (str): 图像文件类型，默认为 '.tif'
            transform (callable, optional): 可选的图像变换
            max_frames (int, optional): 最大帧数，用于统一视频长度
            segment_length (int): 每个视频段的帧数
            overlap_ratio (float): 视频段之间的重叠比例 (0-1)
        """
        self.root_dir = root_dir
        self.label_file = label_file
        self.transform = transform
        self.max_frames = max_frames
        self.file_type = file_type
        self.segment_length = segment_length
        self.overlap_ratio = overlap_ratio
        self.step_size = max(1, int(segment_length * (1 - overlap_ratio)))
        
        # 存储视频段信息：(视频路径, 起始帧索引, 结束帧索引, 对应标签)
        self.video_segments = []
        
        # 读取标签文件并创建视频段
        try:
            self._read_labels_and_create_segments()
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
            raise e
    
    def _read_labels_robust(self):
        """健壮地读取标签文件"""
        video_paths = []
        video_labels = []
        
        # 方法1: 按行读取，手动解析
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 分割行，但保留路径完整性
            parts = line.split()
            if len(parts) < 2:
                continue
                
            # 第一个部分是视频路径
            video_path = parts[0]
            # 剩余部分是标签
            labels = [int(x) for x in parts[1:] if x.isdigit() or x in ['0', '1']]
            
            # 处理路径
            if not os.path.isabs(video_path):
                # 如果是相对路径，尝试构建完整路径
                possible_paths = [
                    os.path.join(self.root_dir, os.path.basename(video_path)),
                    os.path.join(self.root_dir, video_path),
                    video_path
                ]
                
                actual_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        actual_path = path
                        break
                
                if actual_path is None:
                    print(f"警告: 无法找到视频路径 {video_path}")
                    continue
                video_path = actual_path
            
            if os.path.exists(video_path):
                video_paths.append(video_path)
                video_labels.append(labels)
            else:
                print(f"警告: 视频路径不存在 {video_path}")
                
        return video_paths, video_labels
    
    def _read_labels_and_create_segments(self):
        """读取标签并创建视频段"""
        video_paths, video_labels = self._read_labels_robust()
        print("video_paths:",video_paths)

        
        # 为每个视频创建段
        for video_path, labels in zip(video_paths, video_labels):
            # 限制最大帧数
            if self.max_frames is not None:
                frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(self.file_type)])
                frame_files = frame_files[:self.max_frames]
                labels = labels[:self.max_frames]
            else:
                frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(self.file_type)])
            
            total_frames = len(frame_files)
            
            # 如果视频帧数少于段长度，作为一个完整段
            if total_frames <= self.segment_length:
                self.video_segments.append((video_path, 0, total_frames, max(labels)))
            else:
                # 创建重叠的视频段
                start_idx = 0
                while start_idx < total_frames:
                    end_idx = min(start_idx + self.segment_length, total_frames)
                    segment_labels = labels[start_idx:end_idx]
                    
                    # 使用段内标签的最大值作为该段的标签
                    # segment_label = max(segment_labels)
                    self.video_segments.append((video_path, start_idx, end_idx, segment_labels))
                    
                    start_idx += self.step_size    
    def __len__(self):
        """返回数据集大小（视频段数量）"""
        return len(self.video_segments)
    
    def __getitem__(self, idx):
        """
        获取指定索引的视频段
        
        参数:
            idx (int): 数据项索引
            
        返回:
            tuple: (video_tensor, labels_tensor)
                - video_tensor: 视频张量 [channels, frames, height, width]
                - labels_tensor: 标签张量 [frames]
        """
        # 获取视频段信息
        video_path, start_frame, end_frame, labels = self.video_segments[idx]
        
        # 获取视频帧文件列表
        try:
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(self.file_type)])
            # print("frame_files:",frame_files)
        except Exception as e:
            print(f"无法读取目录 {video_path}: {e}")
            return self._get_default_segment(labels)
        
        # 限制帧数并获取指定范围的帧
        if self.max_frames is not None:
            frame_files = frame_files[:self.max_frames]
        
        # 获取该段的帧文件
        segment_frame_files = frame_files[start_frame:end_frame]
        
        # 加载视频帧
        frames = []
        for frame_file in segment_frame_files:
            frame_path = os.path.join(video_path, frame_file)

            try:
                # 使用PIL加载图像
                image = Image.open(frame_path)
                
                # 转换为灰度图（如果需要）
                if image.mode != 'L':
                    image = image.convert('L')
                
                # 应用变换
                if self.transform:
                    image = self.transform(image)
                else:
                    # 默认转换为张量并归一化到[0,1]
                    image = transforms.ToTensor()(image)
                
                frames.append(image)
            except Exception as e:
                print(f"加载帧 {frame_path} 时出错: {e}")
                # 如果加载失败，使用零张量替代
                if frames:
                    frames.append(torch.zeros_like(frames[-1]))
                else:
                    frames.append(torch.zeros(1, 256, 256))  # 默认大小
        
        # 处理空帧情况
        if not frames:
            print("-----------警告: 段内没有有效帧-------------")
            return self._get_default_segment(labels)
        else:
            # 堆叠帧张量 [channels, frames, height, width]
            try:
                video_tensor = torch.stack(frames, dim=1)
            except Exception as e:
                print(f"堆叠帧时出错: {e}")
                return self._get_default_segment(labels)
            
            # 确保标签长度与帧数一致
            # 确保 labels 是一个列表
            if isinstance(labels, int):
                labels = [labels]
            if len(labels) > video_tensor.shape[1]:
                labels = labels[:video_tensor.shape[1]]
            elif len(labels) < video_tensor.shape[1]:
                # 如果标签较少，用0填充
                labels.extend([0] * (video_tensor.shape[1] - len(labels)))
            
            # 转换标签为张量
            try:
                labels_tensor = torch.tensor(labels, dtype=torch.long)  # 改为long类型
            except Exception as e:
                print(f"转换标签为张量时出错: {e}")
                labels_tensor = torch.zeros(video_tensor.shape[1], dtype=torch.long)  # 改为long类型
        
        return video_tensor, labels_tensor
    
    def _get_default_segment(self, labels):
        """获取默认视频段张量"""
        # 如果没有帧，返回零张量
        default_frames = max(len(labels), self.segment_length) if len(labels) > 0 else self.segment_length
        video_tensor = torch.zeros(1, default_frames, 256, 256)
        labels_tensor = torch.zeros(default_frames, dtype=torch.long)  # 改为long类型
        return video_tensor, labels_tensor

def create_train_test_datasets(train_root_dir, test_root_dir, labels_dir, 
                              train_transform=None, test_transform=None, 
                              max_frames=Config.MAX_FRAMES, file_type=Config.FILE_TYPE,
                              segment_length=Config.DATASET_PARAMS['segment_length'], 
                              overlap_ratio=Config.DATASET_PARAMS['overlap_ratio']):
    """
    创建训练和测试数据集（支持视频分段）
    
    参数:
        train_root_dir (str): 训练数据根目录路径
        test_root_dir (str): 测试数据根目录路径
        labels_dir (str): 标签文件目录路径
        train_transform (callable, optional): 训练集图像变换
        test_transform (callable, optional): 测试集图像变换
        max_frames (int, optional): 最大帧数
        file_type (str): 图像文件类型
        segment_length (int): 每个视频段的帧数
        overlap_ratio (float): 视频段之间的重叠比例
        
    返回:
        tuple: (train_dataset, test_dataset)
    """
    # 训练集标签文件路径
    train_label_file = os.path.join(labels_dir, 'train.csv')
    # 测试集标签文件路径
    test_label_file = os.path.join(labels_dir, 'test.csv')
    
    print(f"训练标签文件: {train_label_file}")
    print(f"测试标签文件: {test_label_file}")
    
    # 检查标签文件是否存在
    if not os.path.exists(train_label_file):
        raise FileNotFoundError(f"训练标签文件不存在: {train_label_file}")
    
    if not os.path.exists(test_label_file):
        raise FileNotFoundError(f"测试标签文件不存在: {test_label_file}")
    
    # 创建训练集
    print("创建训练数据集...")
    train_dataset = VideoAnomalyDataset(
        root_dir=train_root_dir,
        label_file=train_label_file,
        file_type=file_type,
        transform=train_transform,
        max_frames=max_frames,
        segment_length=segment_length,
        overlap_ratio=overlap_ratio
    )
    
    # 创建测试集
    print("创建测试数据集...")
    test_dataset = VideoAnomalyDataset(
        root_dir=test_root_dir,
        label_file=test_label_file,
        file_type=file_type,
        transform=test_transform,
        max_frames=max_frames,
        segment_length=segment_length,
        overlap_ratio=overlap_ratio
    )
    
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=Config.TRAINING_PARAMS['batch_size'], 
                     num_workers=Config.TRAINING_PARAMS['num_workers'], shuffle_train=True):
    """
    创建数据加载器
    
    参数:
        train_dataset (Dataset): 训练数据集
        test_dataset (Dataset): 测试数据集
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        shuffle_train (bool): 是否打乱训练数据
        
    返回:
        tuple: (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # 自定义批处理函数
        drop_last=False  # 不丢弃最后一个不完整的批次
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # 自定义批处理函数
        drop_last=False
    )
    
    return train_loader, test_loader

def custom_collate_fn(batch):
    """
    自定义批处理函数，处理不同长度的视频序列
    
    参数:
        batch (list): 批次数据列表
        
    返回:
        tuple: (padded_videos, padded_labels)
    """
    # 分离视频和标签
    videos, labels = zip(*batch)
    
    # 获取批次中视频的最大帧数
    max_frames = max(video.shape[1] for video in videos)
    
    # 填充视频和标签到相同长度
    padded_videos = []
    padded_labels = []
    
    for video, label in zip(videos, labels):
        current_frames = video.shape[1]
        
        if current_frames < max_frames:
            # 填充视频帧
            padding_frames = max_frames - current_frames
            padding = torch.zeros(video.shape[0], padding_frames, video.shape[2], video.shape[3])
            padded_video = torch.cat([video, padding], dim=1)
            
            # 填充标签
            padding_labels = torch.zeros(padding_frames)
            padded_label = torch.cat([label, padding_labels], dim=0)
        else:
            padded_video = video
            padded_label = label
        
        padded_videos.append(padded_video)
        padded_labels.append(padded_label)
    
    # 堆叠批次数据
    try:
        batch_videos = torch.stack(padded_videos, dim=0)  # [batch, channels, frames, height, width]
        batch_labels = torch.stack(padded_labels, dim=0)  # [batch, frames]
    except Exception as e:
        print(f"堆叠批次数据时出错: {e}")
        # 如果堆叠失败，返回第一个样本
        batch_videos = padded_videos[0].unsqueeze(0)
        batch_labels = padded_labels[0].unsqueeze(0)
    
    return batch_videos, batch_labels

# 数据集可视化函数
def visualize_dataset_sample(dataset, sample_idx=0):
    """
    可视化数据集中的样本
    
    参数:
        dataset (VideoAnomalyDataset): 数据集
        sample_idx (int): 样本索引
    """
    try:
        import matplotlib.pyplot as plt
        
        video, labels = dataset[sample_idx]
        
        print(f"视频张量形状: {video.shape}")
        print(f"标签张量形状: {labels.shape}")
        print(f"异常帧数量: {torch.sum(labels).item()}")
        print(f"异常帧索引: {torch.where(labels == 1)[0].tolist()}")
        
        # 可视化前几帧
        num_frames_to_show = min(6, video.shape[1])
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(num_frames_to_show):
            frame = video[0, i].numpy()  # 取第一通道
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i+1}\nLabel: {int(labels[i])}')
            axes[i].axis('off')
        
        plt.suptitle(f'Dataset Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for visualization")
    except Exception as e:
        print(f"可视化时出错: {e}")

# 测试函数
def test_dataset_loading():
    """测试数据集加载"""
    # 请根据实际路径修改以下路径
    train_root_dir = Config.TRAIN_ROOT_DIR  # 修改为实际训练数据路径
    test_root_dir = Config.TEST_ROOT_DIR    # 修改为实际测试数据路径
    labels_dir = Config.LABELS_DIR      # 修改为实际标签文件路径
    
    # 检查路径是否存在
    print("检查路径是否存在:")
    print(f"训练根目录: {os.path.exists(train_root_dir)}")
    print(f"测试根目录: {os.path.exists(test_root_dir)}")
    print(f"标签目录: {os.path.exists(labels_dir)}")
    
    if not all([os.path.exists(train_root_dir), os.path.exists(test_root_dir), os.path.exists(labels_dir)]):
        print("警告: 部分路径不存在，请检查路径设置")
        return
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    try:
        # 创建数据集
        train_dataset, test_dataset = create_train_test_datasets(
            train_root_dir=train_root_dir,
            test_root_dir=test_root_dir,
            labels_dir=labels_dir,
            train_transform=transform,
            test_transform=transform,
            max_frames=Config.MAX_FRAMES,
            file_type=Config.FILE_TYPE,
            segment_length=Config.DATASET_PARAMS['segment_length'],
            overlap_ratio=Config.DATASET_PARAMS['overlap_ratio']
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        if len(train_dataset) > 0:
            print("\n测试加载第一个训练样本:")
            video, labels = train_dataset[0]
            print(f"  视频张量形状: {video.shape}")
            print(f"  标签张量形状: {labels.shape}")
            
            # 可视化样本
            visualize_dataset_sample(train_dataset, sample_idx=0)
        
        # 创建数据加载器
        train_loader, test_loader = get_data_loaders(
            train_dataset, 
            test_dataset, 
            batch_size=Config.TRAINING_PARAMS['batch_size'], 
            num_workers=Config.TRAINING_PARAMS['num_workers']
        )
        
        print(f"\n训练加载器批次数量: {len(train_loader)}")
        print(f"测试加载器批次数量: {len(test_loader)}")
        
        # 测试数据加载
        # print("\n测试数据加载器:")
        # for batch_idx, (videos, labels) in enumerate(train_loader):
        #     print(f"批次 {batch_idx}:")
        #     print(f"  视频张量形状: {videos.shape}")  # [batch, channels, frames, height, width]
        #     print(f"  标签张量形状: {labels.shape}")  # [batch, frames]
            
        #     # 测试与WaveletSTGNN的集成
        #     print("  可与WaveletSTGNN集成使用")
            
        #     if batch_idx >= 2:  # 只测试前几个批次
        #         break
        all_labels=[]
        for batch in test_loader:
            _, labels = batch
            # 统计每个视频段中出现次数最多的标签
            all_labels.extend(labels.tolist())

        # 将列表转换为 NumPy 数组
        all_labels = np.array(all_labels).flatten()  # 确保是一维数组
        print(f"all_labels length: {len(all_labels)}, 0:{sum(all_labels==0)}, 1:{sum(all_labels==1)}")
                
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

# 主函数
if __name__ == "__main__":
    print("视频异常检测数据集加载器")
    print("=" * 40)
    
    # 运行测试
    test_dataset_loading()
    
    print("\n使用说明:")
    print("1. 请根据实际数据集路径修改代码中的路径")
    print("2. 确保标签文件格式正确")
    print("3. 确保图像文件存在且可读")