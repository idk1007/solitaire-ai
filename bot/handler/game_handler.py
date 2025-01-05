import os
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

        print("\n開始執行遊戲邏輯...")
        print(f"預計執行最多 {max_moves} 次移動")

        while moves < max_moves:
            moves += 1
            print(f"\n=== 第 {moves} 輪操作 ===")

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
            print("\n檢查是否可移動到基礎牌堆...")
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

            print(f"\n第 {moves} 輪操作完成")
            print("等待下一輪...")
            random_sleep(1, 2)

        print("\n遊戲操作完成！")
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
