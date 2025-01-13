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

def clear_screen():
        """
        清除終端畫面
        根據作業系統使用不同的清屏指令
        """
        if sys.platform.startswith('win'):  # Windows
            os.system('cls')
        else:  # Mac/Linux
            os.system('clear')
        pass

def print_banner():
    # ... 保留現有的 print_banner 函數 ...
    pass

# 用新版本替換原有的 main 函數
def main():
    print("程序开始执行...")  # 调试输出
    try:
        game = GameHandler()
        print("GameHandler 已创建")  # 调试输出
        
        if game.start():
            try:
                # 添加測試選項
                print("\n請選擇操作：")
                print("1. 開始正常遊戲")
                print("2. 測試撲克牌識別")
                choice = input("請輸入選項（1或2）: ").strip()
                
                if choice == "1":
                    # 原有的遊戲流程
                    game.handle_permission_dialog()
                    time.sleep(1)
                    game.start_game()
                    time.sleep(2)
                    game.play_game()
                elif choice == "2":
                    # 執行測試
                    game.test_card_detection()
                else:
                    print("無效的選項！")
                
            except Exception as e:
                print(f"执行过程中发生错误: {str(e)}")
            finally:
                game.stop()
        else:
            print("游戏处理器启动失败")
    except Exception as e:
        print(f"程序执行过程中发生错误: {str(e)}")



if __name__ == "__main__":
    clear_screen()
    print_banner()
    
    try:
        game = GameHandler()
        if game.start():
            # 執行測試
            game.test_card_detection()
        game.stop()
    except KeyboardInterrupt:
        print("\n\n程式被使用者中斷")
    except Exception as e:
        print(f"\n程式發生錯誤: {e}")
    finally:
        print("\n程式結束")
