# game_handler.py 的修改版本
import os
import sys
import time
import torch
from Solitaire import SolitaireAI, SolitaireEnv

class GameHandler:
    def __init__(self):
        self.emulator = EmulatorConnector()
        self.ai = SolitaireAI()  # 創建AI實例
        self.env = SolitaireEnv()  # 創建環境實例
        # 創建模板目錄
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        
        # 加載預訓練的模型（如果有的話）
        model_path = "trained_models/solitaire_model.pt"
        if os.path.exists(model_path):
            self.ai.load_model(model_path)
            print("已加載預訓練模型")

    def get_game_state(self):
        """
        透過截圖獲取當前遊戲狀態
        返回轉換後的狀態向量
        """
        screenshot = self.take_screenshot("current_state.png")
        # TODO: 實現圖像識別來獲取遊戲狀態
        # 這裡需要添加將截圖轉換為遊戲狀態的邏輯
        return self.env.reset()  # 暫時返回初始狀態

    def execute_action(self, action):
        """
        執行AI決定的動作
        """
        if action is None:
            return False

        if action[0] == 'stock':
            # 點擊牌堆
            return self.click(10, 20)  # 調整座標到實際牌堆位置
        
        elif action[0] == 'waste':
            if action[1] == 'tableau':
                # 從waste移動到tableau
                source_x, source_y = 20, 20  # waste位置
                dest_x, dest_y = self.get_card_position(action[2])
                return self.click(source_x, source_y) and self.click(dest_x, dest_y)
            
            elif action[1] == 'foundation':
                # 從waste移動到foundation
                source_x, source_y = 20, 20  # waste位置
                dest_x, dest_y = self.get_foundation_position(action[2])
                return self.click(source_x, source_y) and self.click(dest_x, dest_y)
        
        elif action[0] == 'tableau':
            # 從tableau移動
            source_x, source_y = self.get_card_position(action[1])
            if action[3] == 'tableau':
                dest_x, dest_y = self.get_card_position(action[4])
            else:  # foundation
                dest_x, dest_y = self.get_foundation_position(action[4])
            return self.click(source_x, source_y) and self.click(dest_x, dest_y)

        return False

    def play_game(self):
        """
        使用AI玩遊戲的主要邏輯
        """
        print("\n開始執行AI遊戲邏輯...")
        moves = 0
        max_moves = 100

        while moves < max_moves:
            moves += 1
            print(f"\n=== 第 {moves} 輪操作 ===")

            # 獲取當前遊戲狀態
            state = self.get_game_state()
            
            # 讓AI決定下一步動作
            action = self.ai.act(state)
            if action is None:
                print("沒有可用的移動")
                break

            # 執行動作
            print(f"執行動作: {action}")
            success = self.execute_action(action)
            if not success:
                print("動作執行失敗")
                continue

            # 等待動畫完成
            random_sleep(0.5, 1)

            # 檢查遊戲是否結束
            # TODO: 實現遊戲結束檢測邏輯

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
            
            # 開始AI遊戲
            game.play_game()
            
        finally:
            game.stop()

if __name__ == "__main__":
    main()