import subprocess
import time
import sys
import os
import cv2
import numpy as np

# 添加父目錄到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ADB_PATH, DEFAULT_PORT, SCREENSHOT_DIR
from utils.utils import find_template

class EmulatorConnector:
    def __init__(self, adb_path=ADB_PATH, port=DEFAULT_PORT):
        self.adb_path = adb_path
        self.port = port
        self.device_id = None

        # 創建截圖目錄
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    def connect(self):
        """
        連接到雷電模擬器
        """
        try:
            # 啟動 ADB server
            subprocess.run([self.adb_path, 'start-server'], check=True)
            
            # 連接到模擬器
            result = subprocess.run(
                [self.adb_path, 'connect', f'127.0.0.1:{self.port}'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if 'connected' in result.stdout.lower():
                print(f"成功連接到模擬器 (port: {self.port})")
                self.device_id = f'127.0.0.1:{self.port}'
                return True
            else:
                print("連接失敗")
                return False
        except subprocess.CalledProcessError as e:
            print(f"連接過程發生錯誤: {e}")
            return False

    def disconnect(self):
        """
        斷開與模擬器的連接
        """
        if self.device_id:
            try:
                subprocess.run([self.adb_path, 'disconnect', self.device_id], check=True)
                print("已斷開與模擬器的連接")
            except subprocess.CalledProcessError as e:
                print(f"斷開連接時發生錯誤: {e}")

    def tap(self, x, y):
        """
        點擊指定座標
        """
        try:
            subprocess.run([self.adb_path, '-s', self.device_id, 'shell', 'input', 'tap', str(x), str(y)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def swipe(self, start_x, start_y, end_x, end_y, duration=1000):
        """
        滑動操作
        duration: 滑動持續時間(毫秒)
        """
        try:
            subprocess.run([
                self.adb_path, '-s', self.device_id, 'shell', 'input', 'swipe',
                str(start_x), str(start_y), str(end_x), str(end_y), str(duration)
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def screenshot(self, output_path=None):
        """
        截取螢幕截圖
        """
        if output_path is None:
            output_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.png")
        try:
            # 先將截圖保存到模擬器中
            subprocess.run([
                self.adb_path, '-s', self.device_id, 'shell', 'screencap', '/sdcard/screenshot.png'
            ], check=True)
            
            # 將截圖拉取到電腦
            subprocess.run([
                self.adb_path, '-s', self.device_id, 'pull', '/sdcard/screenshot.png', output_path
            ], check=True)
            return output_path
        except subprocess.CalledProcessError:
            return None

    def find_and_click(self, template_path, threshold=0.8):
        """
        查找並點擊指定圖片
        """
        screenshot_path = self.screenshot()
        if not screenshot_path:
            return False

        coordinates = find_template(screenshot_path, template_path, threshold)
        if coordinates:
            return self.tap(coordinates[0], coordinates[1])
        return False
