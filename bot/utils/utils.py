import os
import time
import random
import cv2
import numpy as np

# 定义常数
TEMPLATE_DIR = "templates"  # 模板图片目录
SCREENSHOT_DIR = "screenshots"  # 截图保存目录

def random_sleep(min_time=0.5, max_time=1.5):
    """
    随机休眠一段时间
    
    Args:
        min_time (float): 最小休眠时间(秒)
        max_time (float): 最大休眠时间(秒)
    """
    sleep_time = random.uniform(min_time, max_time)
    time.sleep(sleep_time)

def ensure_dir_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_template(image, template_path, threshold=0.8):
    """
    在图片中查找模板
    
    Args:
        image: 要搜索的图片（numpy数组）
        template_path: 模板图片的路径
        threshold: 匹配阈值（0-1之间）
    
    Returns:
        tuple: (中心x坐标, 中心y坐标) 如果找到匹配
        None: 如果没有找到匹配
    """
    # 确保模板文件存在
    if not os.path.exists(template_path):
        print(f"模板文件不存在: {template_path}")
        return None
    
    # 读取模板图片
    template = cv2.imread(template_path)
    if template is None:
        print(f"无法读取模板图片: {template_path}")
        return None
    
    # 获取模板尺寸
    h, w = template.shape[:2]
    
    # 执行模板匹配
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 如果匹配度超过阈值
    if max_val >= threshold:
        # 计算中心点坐标
        center_x = max_loc[0] + w//2
        center_y = max_loc[1] + h//2
        return (center_x, center_y)
    
    return None

def get_timestamp():
    """
    获取当前时间戳字符串
    
    Returns:
        str: 格式化的时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S")

def log_message(message, level="INFO"):
    """
    记录日志消息
    
    Args:
        message (str): 日志消息
        level (str): 日志级别
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] {timestamp} - {message}")

# 创建必要的目录
ensure_dir_exists(TEMPLATE_DIR)
ensure_dir_exists(SCREENSHOT_DIR)