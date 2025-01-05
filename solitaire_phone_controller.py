import cv2
import numpy as np
import time
from phone_controller import PhoneController
from card_detection import detect_cards, recognize_card, draw_debug_info

class SolitairePhoneController:
    def __init__(self, phone_ip="127.0.0.1"):  # 預設使用模擬器
        self.phone = PhoneController(phone_ip)
        self.game_area = None
        self.card_positions = {}
        
    def initialize(self):
        """初始化並連接手機"""
        if not self.phone.connect():
            return False
            
        # 獲取第一張截圖來校準遊戲區域
        self.calibrate_game_area()
        return True
        
    def calibrate_game_area(self):
        """校準遊戲區域"""
        screen = self.phone.get_screen_capture()
        if screen is None:
            return False
            
        # 這裡需要根據實際的遊戲UI來調整
        height, width = screen.shape[:2]
        self.game_area = {
            'top': int(height * 0.1),
            'bottom': int(height * 0.9),
            'left': int(width * 0.05),
            'right': int(width * 0.95)
        }
        
        # 保存調試圖像
        cv2.rectangle(screen, 
                     (self.game_area['left'], self.game_area['top']),
                     (self.game_area['right'], self.game_area['bottom']),
                     (0, 255, 0), 2)
        cv2.imwrite('game_area.png', screen)
        
        return True
        
    def detect_game_state(self):
        """檢測當前遊戲狀態"""
        screen = self.phone.get_screen_capture()
        if screen is None:
            return None
            
        # 裁剪遊戲區域
        game_screen = screen[
            self.game_area['top']:self.game_area['bottom'],
            self.game_area['left']:self.game_area['right']
        ]
        
        # 檢測卡牌
        success, cards = detect_cards(game_screen)
        if not success:
            return None
            
        # 保存調試圖像
        debug_image = draw_debug_info(game_screen, cards)
        cv2.imwrite('detected_cards.png', debug_image)
        
        # 更新卡牌位置信息
        self.card_positions = {}
        for card in cards:
            pos = card['position']
            card_info = recognize_card(card['image'])
            if card_info is not None:
                self.card_positions[str(card_info)] = (
                    pos[0] + self.game_area['left'],
                    pos[1] + self.game_area['top']
                )
                
        return {
            'cards': cards,
            'positions': self.card_positions
        }
        
    def make_move(self, from_pos, to_pos):
        """執行移動"""
        # 將相對位置轉換為絕對位置
        start_x, start_y = from_pos
        end_x, end_y = to_pos
        
        # 執行拖動操作
        return self.phone.swipe(start_x, start_y, end_x, end_y, 300)
        
    def play_game(self):
        """自動玩遊戲"""
        while True:
            # 檢測當前遊戲狀態
            game_state = self.detect_game_state()
            if game_state is None:
                print("Failed to detect game state")
                break
                
            # 顯示檢測結果
            print(f"Detected {len(game_state['cards'])} cards")
            print(f"Recognized positions: {len(game_state['positions'])}")
            
            # 暫停一下，避免操作太快
            time.sleep(0.5)
            
    def cleanup(self):
        """清理並斷開連接"""
        self.phone.disconnect()

def test_basic_functions(controller):
    """測試基本功能"""
    print("Testing basic functions...")
    
    # 測試截圖
    print("Testing screen capture...")
    screen = controller.phone.get_screen_capture()
    if screen is not None:
        cv2.imwrite('test_screenshot.png', screen)
        print("Screenshot saved as test_screenshot.png")
    else:
        print("Failed to capture screen")
        return False
        
    # 測試點擊
    print("Testing tap...")
    if not controller.phone.tap(100, 100):
        print("Failed to perform tap")
        return False
    time.sleep(1)
    
    # 測試滑動
    print("Testing swipe...")
    if not controller.phone.swipe(100, 100, 300, 300, 300):
        print("Failed to perform swipe")
        return False
    time.sleep(1)
    
    print("Basic function tests completed successfully")
    return True

# 在 main 函數中添加更多的調試信息
def main():
    controller = SolitairePhoneController("192.168.0.184")
    
    try:
        print("Initializing controller...")
        if controller.initialize():
            print("Controller initialized successfully")
            
            # 保存初始截圖用於調試
            screen = controller.phone.get_screen_capture()
            if screen is not None:
                cv2.imwrite('initial_screen.png', screen)
                print("Initial screen saved")
            
            # 執行基本功能測試
            if test_basic_functions(controller):
                print("Basic tests passed, starting game...")
                # 在開始遊戲前暫停，檢查截圖結果
                input("Press Enter to continue...")
                controller.play_game()
            else:
                print("Basic tests failed")
        else:
            print("Failed to initialize controller")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        print("Cleaning up...")
        controller.cleanup()
        print("Done")

if __name__ == "__main__":
    main()