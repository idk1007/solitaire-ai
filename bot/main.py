import os
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
    print("\n開始自動遊戲...")
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
            print("\n感謝使用，程式結束")
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
        
        print("\n按 Enter 鍵繼續...")
        input()
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程式被使用者中斷")
    except Exception as e:
        print(f"\n程式發生錯誤: {e}")
    finally:
        print("\n程式結束")
