import os
import logging
from prettytable import PrettyTable
# 配置日志
def setup_logging():
    """
    设置日志配置
    """
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建logger
    logger = logging.getLogger('anomaly_detection')
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers（避免重复添加）
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件handler，使用 'a' 模式以追加到原有日志文件而不是覆盖
    # 修改开始：使用追加模式并确保日志被写入
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    
    # 禁用日志缓冲，确保日志立即写入
    file_handler.stream.flush()
    # 修改结束
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def printGrad(model):
    table = PrettyTable()
    table.field_names = ["Parameter Name", "Shape", "Mean", "Std", "L2-Norm", "Has Grad"]
    for name, param in model.named_parameters():
        g = param.grad
        if g is None:
            table.add_row([name, str(tuple(param.shape)), "—", "—", "—", "False"])
        else:
            table.add_row([
                name,
                str(tuple(param.shape)),
                f"{g.mean().item():+.4e}",
                f"{g.std().item():+.4e}",
                f"{g.norm().item():+.4e}",
                "True"
            ])

    # 可选：左对齐第一列
    table.align["Parameter Name"] = "l"
    print(table)