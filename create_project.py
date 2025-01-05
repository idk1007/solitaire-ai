import os
import shutil


def create_directories(base_path, structure, current_path=''):
    """
    遞迴創建目錄結構和文件
    
    Args:
        base_path (str): 基礎路徑
        structure (dict): 目錄結構字典
        current_path (str): 當前處理的路徑
    """
    for name, content in structure.items():
        path = os.path.join(base_path, current_path, name)
        
        if isinstance(content, dict):
            # 如果是目錄
            os.makedirs(path, exist_ok=True)
            
            # 遞迴處理子目錄
            for sub_name, sub_content in content.items():
                if isinstance(sub_content, str):
                    # 如果是文件
                    file_path = os.path.join(path, sub_name)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(sub_content)
                elif isinstance(sub_content, dict):
                    # 如果是子目錄，遞迴創建
                    sub_path = os.path.join(current_path, name)
                    create_directories(base_path, {sub_name: sub_content}, sub_path)

def create_project_structure():
    # 定義基礎目錄結構
    structure = {
        'bot': {
            '.vscode': {},
            'config': {
                '__pycache__': {},
                '__init__.py': '',
                'config.py': '''# 雷電模擬器相關設定
ADB_PATH = r"D:\LDPlayer\LDPlayer9\adb.exe"  # 雷電模擬器的 ADB 路徑
DEFAULT_PORT = 5555  # 預設 ADB 連接埠

# 螢幕解析度設定
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# 操作延遲設定(秒)
CLICK_DELAY = 0.5
SWIPE_DELAY = 1.0

# 圖像識別設定
TEMPLATE_DIR = "templates"  # 模板圖片目錄
SCREENSHOT_DIR = "screenshots"  # 截圖保存目錄
MATCH_THRESHOLD = 0.8  # 圖像匹配閾值
'''
            },
            'connector': {
                '__pycache__': {},
                '__init__.py': '',
                'emulator_connector.py': '''import subprocess
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
'''
            },
            'handler': {
                '__pycache__': {},
                '__init__.py': '',
                'game_handler.py': '''import os
import sys
import time

# 添加父目錄到系統路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connector.emulator_connector import EmulatorConnector
from utils.utils import random_sleep, calculate_coordinates
from config.config import TEMPLATE_DIR

class GameHandler:
    def __init__(self):
        self.emulator = EmulatorConnector()
        # 創建模板目錄
        os.makedirs(TEMPLATE_DIR, exist_ok=True)

    def clear_terminal(self):
        """
        清除終端畫面
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def start(self):
        """
        啟動遊戲控制
        """
        self.clear_terminal()  # 在啟動時清除終端
        if not self.emulator.connect():
            print("無法連接到模擬器")
            return False
        print("成功連接到模擬器")
        return True

    def stop(self):
        """
        停止遊戲控制
        """
        self.emulator.disconnect()

    def click(self, x_percent, y_percent):
        """
        點擊指定位置(使用百分比座標)
        """
        x, y = calculate_coordinates(x_percent, y_percent)
        success = self.emulator.tap(x, y)
        if success:
            random_sleep()
        return success

    def swipe(self, start_x_percent, start_y_percent, end_x_percent, end_y_percent, duration=1000):
        """
        滑動操作(使用百分比座標)
        """
        start_x, start_y = calculate_coordinates(start_x_percent, start_y_percent)
        end_x, end_y = calculate_coordinates(end_x_percent, end_y_percent)
        success = self.emulator.swipe(start_x, start_y, end_x, end_y, duration)
        if success:
            random_sleep()
        return success

    def take_screenshot(self, filename):
        """
        截取遊戲畫面
        """
        return self.emulator.screenshot(filename)

    def handle_permission_dialog(self):
        """
        處理權限對話框
        """
        return self.click(75, 85)  # 點擊「允許」按鈕

    def start_game(self):
        """
        開始遊戲
        """
        return self.click(50, 85)  # 點擊「開始遊戲」按鈕

    def get_card_position(self, column):
        """
        獲取指定列的牌堆位置
        column: 0-6 (從左到右的7列牌堆)
        """
        base_x = 15  # 最左邊列的x座標
        column_width = 10  # 列之間的間距
        x = base_x + (column * column_width)
        y = 50  # 牌堆大約在螢幕中間高度
        return x, y

    def get_foundation_position(self, suit_index):
        """
        獲取右上方基礎牌堆的位置
        suit_index: 0-3 (四種花色的位置)
        """
        base_x = 55  # 最左邊基礎牌堆的x座標
        x = base_x + (suit_index * 10)
        y = 20  # 基礎牌堆在螢幕上方
        return x, y

    def click_hint(self):
        """
        點擊提示按鈕
        """
        return self.click(50, 90)  # 提示按鈕在螢幕下方中央

    def click_undo(self):
        """
        點擊回上一步按鈕
        """
        return self.click(80, 90)  # 回上一步按鈕在螢幕下方右側

    def play_game(self):
        """
        遊戲主要邏輯
        """
        max_moves = 100  # 最大移動次數
        moves = 0

        print("\\n開始執行遊戲邏輯...")
        print(f"預計執行最多 {max_moves} 次移動")

        while moves < max_moves:
            moves += 1
            print(f"\\n=== 第 {moves} 輪操作 ===")

            # 點擊提示按鈕
            print("點擊提示按鈕...")
            self.click_hint()
            random_sleep(1, 1.5)

            # 嘗試點擊建議的牌
            print("嘗試點擊各列牌堆...")
            for column in range(7):
                print(f"  檢查第 {column + 1} 列...")
                x, y = self.get_card_position(column)
                self.click(x, y)
                random_sleep(0.5, 1)

            # 檢查是否有牌可以移動到基礎牌堆
            print("\\n檢查是否可移動到基礎牌堆...")
            for column in range(7):
                print(f"  檢查第 {column + 1} 列的牌...")
                x, y = self.get_card_position(column)
                self.click(x, y)
                random_sleep(0.5, 1)

                for suit in range(4):
                    print(f"    嘗試移動到第 {suit + 1} 個基礎牌堆")
                    fx, fy = self.get_foundation_position(suit)
                    self.click(fx, fy)
                    random_sleep(0.5, 1)

            print(f"\\n第 {moves} 輪操作完成")
            print("等待下一輪...")
            random_sleep(1, 2)

        print("\\n遊戲操作完成！")
        print(f"總共執行了 {moves} 輪操作")

def main():
    game = GameHandler()
    if game.start():
        try:
            # 處理權限對話框
            game.handle_permission_dialog()
            random_sleep(1, 2)
            
            # 開始遊戲
            game.start_game()
            random_sleep(2, 3)
            
            # 開始玩遊戲
            game.play_game()
            
        finally:
            game.stop()

if __name__ == "__main__":
    main()
'''
            },
            'screenshots': {},
            'templates': {},
            'utils': {
                '__pycache__': {},
                '__init__.py': '',
                'utils.py': '''import time
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
'''
            }
        }
    }
    # 將 main.py 添加到結構中
    structure['bot']['main.py'] = '''import os
import sys
import time
from handler.game_handler import GameHandler
from utils.utils import random_sleep

def print_banner():
    """
    顯示程式啟動橫幅
    """
    banner = """
    ====================================
    LINE 接龍高手 自動化程式
    版本: 1.0
    ====================================
    """
    print(banner)

def print_menu():
    """
    顯示主選單
    """
    menu = """
    請選擇操作:
    1. 開始自動遊戲
    2. 截圖
    3. 測試連接
    0. 退出
    """
    print(menu)

def handle_auto_play(game):
    """
    處理自動遊戲流程
    """
    print("\\n開始自動遊戲...")
    try:
        # 處理權限對話框
        game.handle_permission_dialog()
        random_sleep(1, 2)
        
        # 開始遊戲
        game.start_game()
        random_sleep(2, 3)
        
        # 開始玩遊戲
        game.play_game()
        
    except Exception as e:
        print(f"遊戲過程中發生錯誤: {e}")

def handle_screenshot(game):
    """
    處理截圖功能
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    if game.take_screenshot(filename):
        print(f"截圖已保存: {filename}")
    else:
        print("截圖失敗")

def handle_test_connection(game):
    """
    測試與模擬器的連接
    """
    if game.start():
        print("連接測試成功")
        game.stop()
    else:
        print("連接測試失敗")

def main():
    # 清除終端
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 顯示程式橫幅
    print_banner()
    
    # 創建遊戲處理器實例
    game = GameHandler()
    
    while True:
        print_menu()
        choice = input("請輸入選項: ").strip()
        
        if choice == '0':
            print("\\n感謝使用，程式結束")
            break
            
        elif choice == '1':
            if game.start():
                try:
                    handle_auto_play(game)
                finally:
                    game.stop()
            else:
                print("無法連接到模擬器，請檢查模擬器是否正常運行")
                
        elif choice == '2':
            if game.start():
                try:
                    handle_screenshot(game)
                finally:
                    game.stop()
            else:
                print("無法連接到模擬器，請檢查模擬器是否正常運行")
                
        elif choice == '3':
            handle_test_connection(game)
            
        else:
            print("無效的選項，請重新選擇")
        
        print("\\n按 Enter 鍵繼續...")
        input()
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n程式被使用者中斷")
    except Exception as e:
        print(f"\\n程式發生錯誤: {e}")
    finally:
        print("\\n程式結束")
'''

    # 獲取當前目錄
    current_dir = os.getcwd()
    
    # 創建專案結構
    create_directories(current_dir, structure)
    
    print("專案結構已成功創建！")
    print("目錄結構:")
    for root, dirs, files in os.walk('bot'):
        level = root.replace('bot', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    create_project_structure()