import os
import sys
import time
import cv2
import numpy as np

# 獲取當前文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
# 獲取專案根目錄
project_root = os.path.dirname(current_dir)
# 將專案根目錄添加到 Python 路徑
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import (
    get_timestamp, 
    detect_cards, 
    filter_overlapping_cards, 
    load_card_templates,
    TEMPLATE_DIR,
    SCREENSHOT_DIR,
    CARDS_TEMPLATE_DIR
)
from connector.emulator_connector import EmulatorConnector

class GameHandler:
    def __init__(self):
        self.emulator = EmulatorConnector()
        self.card_templates = load_card_templates()
        # 創建必要的目錄
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        os.makedirs(CARDS_TEMPLATE_DIR, exist_ok=True)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        
        # 設定螢幕尺寸（可以根據實際情況調整）
        self.screen_width = 1920  # 預設寬度
        self.screen_height = 1080  # 預設高度

    def start(self):
        """
        啟動遊戲控制
        """
        print("正在啟動遊戲控制...")
        try:
            # 嘗試連接模擬器
            if not hasattr(self.emulator, 'connect'):
                print("模擬器連接器未正確初始化")
                return False
                
            if not self.emulator.connect():
                print("無法連接到模擬器")
                return False
                
            print("成功連接到模擬器")
            return True
            
        except Exception as e:
            print(f"啟動過程發生錯誤: {str(e)}")
            return False

    def stop(self):
        """
        停止遊戲控制
        """
        try:
            if hasattr(self.emulator, 'disconnect'):
                self.emulator.disconnect()
            print("已停止遊戲控制")
        except Exception as e:
            print(f"停止過程發生錯誤: {str(e)}")

    def take_screenshot(self, filename):
        """
        截取遊戲畫面
        """
        try:
            if not self.emulator or not hasattr(self.emulator, 'screenshot'):
                print("模擬器未正確初始化")
                return False
                
            return self.emulator.screenshot(filename)
        except Exception as e:
            print(f"截圖時發生錯誤: {str(e)}")
            return False

    def click(self, x_percent, y_percent):
        """
        點擊指定位置(使用百分比座標)
        """
        try:
            x = int(self.screen_width * x_percent / 100)
            y = int(self.screen_height * y_percent / 100)
            return self.emulator.tap(x, y)
        except Exception as e:
            print(f"點擊操作時發生錯誤: {str(e)}")
            return False

    def test_card_detection(self):
        """
        測試撲克牌識別系統
        """
        print("\n=== 開始測試撲克牌識別系統 ===")
        
        # 截取當前畫面
        timestamp = get_timestamp()
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"test_screen_{timestamp}.png")
        
        print("正在截取遊戲畫面...")
        if not self.take_screenshot(screenshot_path):
            print("截圖失敗！")
            return False
        
        print(f"截圖已保存至: {screenshot_path}")
        
        # 讀取截圖
        print("正在分析畫面...")
        image = cv2.imread(screenshot_path)
        if image is None:
            print("無法讀取截圖！")
            return False
        
        # 檢測卡片
        print("正在檢測撲克牌...")
        cards = detect_cards(image, self.card_templates)
        filtered_cards = filter_overlapping_cards(cards)
        
        # 在截圖上標記檢測結果
        marked_image = image.copy()
        for card in filtered_cards:
            # 在卡片位置畫圓
            cv2.circle(marked_image, 
                      (int(card.position[0]), int(card.position[1])), 
                      20, (0, 255, 0), 2)
            # 添加卡片信息文字
            cv2.putText(marked_image, 
                       f"{card.suit}-{card.value}", 
                       (int(card.position[0] - 30), int(card.position[1] - 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)
        
        # 保存標記後的圖片
        marked_path = os.path.join(SCREENSHOT_DIR, f"marked_{timestamp}.png")
        cv2.imwrite(marked_path, marked_image)
        
        # 輸出檢測結果
        print(f"\n檢測到 {len(filtered_cards)} 張撲克牌:")
        for i, card in enumerate(filtered_cards, 1):
            print(f"{i}. {card}")
        
        print(f"\n標記後的圖片已保存至: {marked_path}")
        print("=== 測試完成 ===\n")
        
        return True
