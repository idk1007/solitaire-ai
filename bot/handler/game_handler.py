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

class CardDetector:
    def __init__(self):
        # 緩存常用參數
        self.binary_threshold = 160
        self.min_area = 2000
        self.max_area = 25000
        self.aspect_ratio = None  # 將由校準過程設置
        self.aspect_ratio_tolerance = 0.3  # 容許誤差範圍
        
        # 預先創建核心
        self.morph_kernel = np.ones((7,7), np.uint8)
    
    def detect(self, image):
        # 使用 NumPy 操作代替循環
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # 批量處理輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 使用向量化操作
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        valid_contours = [cnt for i, cnt in enumerate(contours) 
                         if self.min_area < areas[i] < self.max_area]
        
        return valid_contours
class GameHandler:
    def __init__(self):
        self.emulator = EmulatorConnector()
        
        # 確保目錄存在
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        os.makedirs(CARDS_TEMPLATE_DIR, exist_ok=True)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        
        # 先載入模板
        self.card_templates = self._initialize_templates()
        
        # 設定螢幕尺寸
        self.screen_width = 1920
        self.screen_height = 1080

        #卡片檢測器參數
        self.card_detector = CardDetector()
        self.card_params = None  # 用於存儲校準後的卡片參數

    def simplified_card_detection(self, image):
        """
        使用校準後的參數進行卡片檢測
        """
        # 1. 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. 應用高斯模糊減少噪點
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. 自適應二值化
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # 4. 形態學操作
        # 先進行膨脹操作
        dilated = cv2.dilate(binary, self.card_detector.morph_kernel, iterations=1)
        # 再進行閉運算清理噪點
        binary = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, self.card_detector.morph_kernel)
        
        # 5. 輪廓檢測
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. 根據面積和長寬比過濾
        cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.card_detector.min_area < area < self.card_detector.max_area:
                # 使用最小外接矩形來獲得更準確的寬高比
                rect = cv2.minAreaRect(contour)
                width = rect[1][0]
                height = rect[1][1]
                
                # 確保寬度大於高度
                if width < height:
                    width, height = height, width
                    
                aspect = width / height if height != 0 else 0
                
                # 使用校準後的長寬比進行比較
                if self.card_detector.aspect_ratio is not None:
                    target_ratio = self.card_detector.aspect_ratio
                    tolerance = self.card_detector.aspect_ratio_tolerance
                    if abs(aspect - target_ratio) <= tolerance:
                        # 計算卡片中心點
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            cards.append({
                                'center': (center_x, center_y),
                                'width': width,
                                'height': height,
                                'area': area,
                                'aspect_ratio': aspect,
                                'contour': contour
                            })
        
        # 保存調試圖像而不是直接顯示
        timestamp = get_timestamp()
        debug_dir = os.path.join(SCREENSHOT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存二值化圖像
        binary_path = os.path.join(debug_dir, f"binary_{timestamp}.png")
        cv2.imwrite(binary_path, binary)
        
        # 保存帶有檢測結果的調試圖像
        debug_image = image.copy()
        cv2.drawContours(debug_image, [card['contour'] for card in cards], -1, (0, 255, 0), 2)
        for card in cards:
            cv2.circle(debug_image, card['center'], 5, (0, 0, 255), -1)
        
        debug_path = os.path.join(debug_dir, f"debug_{timestamp}.png")
        cv2.imwrite(debug_path, debug_image)
        
        print(f"\n調試圖像已保存：")
        print(f"二值化圖像：{binary_path}")
        print(f"檢測結果：{debug_path}")
        
        # 轉換為簡單的坐標列表
        card_centers = [(card['center'][0], card['center'][1]) for card in cards]
        
        # 輸出調試信息
        if cards:
            print("\n檢測到的卡片詳細信息：")
            for i, card in enumerate(cards, 1):
                print(f"卡片 #{i}:")
                print(f"  中心點: {card['center']}")
                print(f"  尺寸: {card['width']:.1f}x{card['height']:.1f}")
                print(f"  面積: {card['area']:.1f}")
                print(f"  長寬比: {card['aspect_ratio']:.3f}")
        else:
            print("\n未檢測到卡片，請檢查以下可能的問題：")
            print("1. 圖像亮度是否足夠")
            print("2. 卡片是否有足夠的對比度")
            print("3. 當前參數設置：")
            print(f"   - 二值化閾值: {self.card_detector.binary_threshold}")
            print(f"   - 面積範圍: {self.card_detector.min_area} - {self.card_detector.max_area}")
            print(f"   - 長寬比容許誤差: {self.card_detector.aspect_ratio_tolerance}")
        
        return card_centers

    def calibrate_card_size(self):
        """
        校準卡片大小並計算標準長寬比
        """
        print("\n=== 開始校準卡片尺寸 ===")
        print("請在遊戲中選擇一張清晰可見的卡片...")
        
        # 拍攝一張只有一張卡的照片
        screenshot = self.take_screenshot("calibration.png")
        image = cv2.imread("calibration.png")
        
        if image is None:
            print("無法讀取校準圖片！")
            return False
        
        # 顯示說明
        print("\n請框選一張完整的卡片（盡量精確）...")
        
        # 手動選擇卡片區域
        roi = cv2.selectROI("Select Card", image)
        cv2.destroyAllWindows()
        
        # 計算尺寸參數
        self.card_width = roi[2]
        self.card_height = roi[3]
        self.card_area = roi[2] * roi[3]
        
        # 計算長寬比
        aspect_ratio = float(self.card_width) / self.card_height
        
        # 更新檢測器的參數
        self.card_detector.aspect_ratio = aspect_ratio
        
        # 根據面積自動調整面積範圍
        self.card_detector.min_area = int(self.card_area * 0.7)  # 允許 70% 的最小面積
        self.card_detector.max_area = int(self.card_area * 1.3)  # 允許 130% 的最大面積
        
        print("\n=== 校準結果 ===")
        print(f"卡片寬度: {self.card_width}")
        print(f"卡片高度: {self.card_height}")
        print(f"卡片面積: {self.card_area}")
        print(f"長寬比: {aspect_ratio:.3f}")
        print(f"面積範圍: {self.card_detector.min_area} - {self.card_detector.max_area}")
        
        return {
            "width": self.card_width,
            "height": self.card_height,
            "area": self.card_area,
            "aspect_ratio": aspect_ratio
        }

    def verify_detection(self, image, cards):
        """
        驗證檢測結果，提供更好的視覺化和互動性
        """
        # 創建一個可以調整的視窗
        window_name = "Card Detection Verification"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)  # 設定合適的視窗大小

        # 複製圖像用於標記
        marked = image.copy()
        
        # 在圖像上標記檢測結果
        for i, (x, y) in enumerate(cards, 1):
            # 畫圓圈標記卡片位置
            cv2.circle(marked, (x, y), 20, (0, 255, 0), 2)
            
            # 添加編號標籤
            cv2.putText(marked, 
                    f"#{i}",
                    (x - 10, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)
            
            # 添加座標信息
            cv2.putText(marked,
                    f"({x}, {y})",
                    (x - 30, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1)

        # 添加操作說明
        height = marked.shape[0]
        instructions = [
            "按鍵說明:",
            "SPACE/ENTER: 確認並繼續",
            "R: 重新檢測",
            "A: 調整參數",
            "ESC: 退出"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(marked,
                    text,
                    (10, height - 20 - (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        while True:
            # 顯示結果
            cv2.imshow(window_name, marked)
            
            # 等待按鍵
            key = cv2.waitKey(0) & 0xFF
            
            if key in [27]:  # ESC
                print("使用者選擇退出")
                break
            elif key in [32, 13]:  # SPACE 或 ENTER
                print("使用者確認檢測結果")
                break
            elif key == ord('r'):  # 重新檢測
                cv2.destroyAllWindows()
                return self.test_card_detection()
            elif key == ord('a'):  # 調整參數
                self.adjust_detection_parameters()
                cv2.destroyAllWindows()
                return self.test_card_detection()

        cv2.destroyAllWindows()

    def adjust_detection_parameters(self):
        """
        調整檢測參數
        """
        print("\n=== 檢測參數調整 ===")
        print(f"當前參數：")
        print(f"1. 二值化閾值: {self.card_detector.binary_threshold}")
        print(f"2. 最小面積: {self.card_detector.min_area}")
        print(f"3. 最大面積: {self.card_detector.max_area}")
        print(f"4. 長寬比容許誤差: {self.card_detector.aspect_ratio_tolerance}")
        
        choice = input("\n請選擇要調整的參數 (1-4)，或按 Enter 退出: ").strip()
        
        if choice == "1":
            try:
                value = int(input("請輸入新的二值化閾值 (0-255): "))
                if 0 <= value <= 255:
                    self.card_detector.binary_threshold = value
                    print(f"已更新二值化閾值為: {value}")
                else:
                    print("無效的值，必須在 0-255 之間")
            except ValueError:
                print("請輸入有效的數字")
        
        elif choice == "2":
            try:
                value = int(input("請輸入新的最小面積: "))
                if value > 0:
                    self.card_detector.min_area = value
                    print(f"已更新最小面積為: {value}")
                else:
                    print("面積必須大於 0")
            except ValueError:
                print("請輸入有效的數字")
        
        elif choice == "3":
            try:
                value = int(input("請輸入新的最大面積: "))
                if value > self.card_detector.min_area:
                    self.card_detector.max_area = value
                    print(f"已更新最大面積為: {value}")
                else:
                    print("最大面積必須大於最小面積")
            except ValueError:
                print("請輸入有效的數字")
        
        elif choice == "4":
            try:
                tolerance = float(input("請輸入新的長寬比容許誤差 (例如 0.3): "))
                if 0 < tolerance < 1:
                    self.card_detector.aspect_ratio_tolerance = tolerance
                    print(f"已更新長寬比容許誤差為: {tolerance}")
                else:
                    print("無效的容許誤差值，必須在 0 到 1 之間")
            except ValueError:
                print("請輸入有效的數字")
        
    def _initialize_templates(self):
        """
        初始化並生成卡片模板
        """
        templates = {}
        
        # 定義花色和數值
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        # 檢查模板目錄是否為空
        if not os.path.exists(CARDS_TEMPLATE_DIR) or not os.listdir(CARDS_TEMPLATE_DIR):
            print("正在生成卡片模板...")
            
            # 創建基本模板圖像
            for suit in suits:
                for value in values:
                    template_path = os.path.join(CARDS_TEMPLATE_DIR, f"{suit}_{value}.png")
                    if not os.path.exists(template_path):
                        # 創建空白圖像
                        template = np.zeros((100, 70, 3), dtype=np.uint8)
                        template.fill(255)  # 白色背景
                        
                        # 繪製花色符號
                        color = (0, 0, 255) if suit in ['hearts', 'diamonds'] else (0, 0, 0)
                        cv2.putText(template, suit[0].upper(), 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, color, 2)
                        
                        # 繪製數值
                        cv2.putText(template, value, 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, color, 2)
                        
                        # 保存模板
                        cv2.imwrite(template_path, template)
                        print(f"已生成模板: {template_path}")
        
        # 載入所有模板
        print("\n正在載入卡片模板...")
        print(f"模板目錄: {CARDS_TEMPLATE_DIR}")
        
        for suit in suits:
            for value in values:
                template_path = os.path.join(CARDS_TEMPLATE_DIR, f"{suit}_{value}.png")
                if os.path.exists(template_path):
                    try:
                        template = cv2.imread(template_path)
                        if template is not None:
                            templates[f"{suit}_{value}"] = template
                            print(f"成功載入: {os.path.basename(template_path)}")
                        else:
                            print(f"無法讀取模板: {template_path}")
                    except Exception as e:
                        print(f"載入模板時發生錯誤 {template_path}: {str(e)}")
        
        print(f"共載入 {len(templates)} 個模板")
        return templates

    def _preprocess_image(self, image):
        """
        簡化的圖像預處理
        """
        # 1. 調整大小（可選）
        scale_percent = 75
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height))
        
        # 2. 簡單的對比度調整
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        return image

    def test_card_detection(self):
        """
        測試撲克牌識別系統
        """
        print("\n=== 開始測試撲克牌識別系統 ===")
        
        # 如果沒有校準過卡片尺寸，先進行校準
        if not self.card_params:
            print("首次運行需要校準卡片尺寸")
            self.card_params = self.calibrate_card_size()
            if not self.card_params:
                return False
        
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
        
        # 顯示原始圖像尺寸
        print(f"圖像尺寸: {image.shape}")
        
        # 圖片預處理
        print("正在預處理圖片...")
        processed = self._preprocess_image(image)
        
        # 使用簡化的檢測方法
        print("正在檢測撲克牌...")
        cards = self.simplified_card_detection(processed)
        
        if not cards:
            print("未檢測到卡片，是否需要調整參數？(Y/N)")
            if input().lower() == 'y':
                self.adjust_detection_parameters()
                return self.test_card_detection()
        
        # 在截圖上標記檢測結果
        marked_image = processed.copy()
        for (x, y) in cards:
            cv2.circle(marked_image, (x, y), 20, (0, 255, 0), 2)
            cv2.putText(marked_image, 
                    f"({x}, {y})",
                    (x - 30, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2)
        
        # 保存標記後的圖片
        marked_path = os.path.join(SCREENSHOT_DIR, f"marked_{timestamp}.png")
        cv2.imwrite(marked_path, marked_image)
        
        # 輸出檢測結果
        print(f"\n檢測到 {len(cards)} 張撲克牌:")
        for i, (x, y) in enumerate(cards, 1):
            print(f"{i}. 位置: ({x}, {y})")
        
        # 驗證結果
        self.verify_detection(processed, cards)
        
        print(f"\n標記後的圖片已保存至: {marked_path}")
        print("=== 測試完成 ===\n")
        
        return True

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

    
    