import time
import random
import cv2
import numpy as np
import os

def random_sleep(min_time=0.5, max_time=1.5):
    """
    隨機等待一段時間，模擬人類操作
    """
    sleep_time = random.uniform(min_time, max_time)
    time.sleep(sleep_time)

def calculate_coordinates(x_percent, y_percent, width=1280, height=720):
    """
    將百分比座標轉換為實際座標
    """
    x = int(width * x_percent / 100)
    y = int(height * y_percent / 100)
    return x, y

def find_template(screenshot, template_path, threshold=0.8):
    """
    在截圖中查找模板圖片
    返回最佳匹配位置的中心點座標
    """
    if not os.path.exists(template_path):
        print(f"模板圖片不存在: {template_path}")
        return None

    # 讀取圖片
    template = cv2.imread(template_path)
    screenshot = cv2.imread(screenshot)

    if template is None or screenshot is None:
        print("圖片讀取失敗")
        return None

    # 進行模板匹配
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        print(f"未找到匹配圖片，相似度: {max_val}")
        return None

    # 計算中心點座標
    w, h = template.shape[1], template.shape[0]
    center_x = max_loc[0] + w//2
    center_y = max_loc[1] + h//2

    return (center_x, center_y)
