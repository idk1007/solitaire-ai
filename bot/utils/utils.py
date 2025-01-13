import os
import time
import random
import cv2
import numpy as np

# 定义常数
TEMPLATE_DIR = "templates"  # 模板图片目录
SCREENSHOT_DIR = "screenshots"  # 截图保存目录
CARDS_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "cards")  # 撲克牌模板目錄

class Card:
    def __init__(self, suit, value, confidence=0.0, position=None):
        self.suit = suit  # 花色: hearts(紅心), diamonds(方塊), clubs(梅花), spades(黑桃)
        self.value = value  # 數值: A,2,3,4,5,6,7,8,9,10,J,Q,K
        self.confidence = confidence  # 識別置信度
        self.position = position  # 卡片在畫面中的位置 (x, y)

    def __str__(self):
        return f"{self.suit}{self.value} at {self.position} (conf: {self.confidence:.2f})"

def random_sleep(min_time=0.5, max_time=1.5):
    """
    随机休眠一段时间
    
    Args:
        min_time (float): 最小休眠时间(秒)
        max_time (float): 最大休眠时间(秒)
    """
    sleep_time = random.uniform(min_time, max_time)
    time.sleep(sleep_time)

def ensure_dir_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_template(image, template_path, threshold=0.8):
    """
    在图片中查找模板
    
    Args:
        image: 要搜索的图片（numpy数组）
        template_path: 模板图片的路径
        threshold: 匹配阈值（0-1之间）
    
    Returns:
        tuple: (中心x坐标, 中心y坐标) 如果找到匹配
        None: 如果没有找到匹配
    """
    # 确保模板文件存在
    if not os.path.exists(template_path):
        print(f"模板文件不存在: {template_path}")
        return None
    
    # 读取模板图片
    template = cv2.imread(template_path)
    if template is None:
        print(f"无法读取模板图片: {template_path}")
        return None
    
    # 获取模板尺寸
    h, w = template.shape[:2]
    
    # 执行模板匹配
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 如果匹配度超过阈值
    if max_val >= threshold:
        # 计算中心点坐标
        center_x = max_loc[0] + w//2
        center_y = max_loc[1] + h//2
        return (center_x, center_y)
    
    return None

def get_timestamp():
    """
    获取当前时间戳字符串
    
    Returns:
        str: 格式化的时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S")

def load_card_templates():
    """
    載入所有撲克牌模板
    Returns:
        dict: 包含所有卡片模板的字典 {(suit, value): template_image}
    """
    templates = {}
    suits = ['hearts', 'diamonds', 'clubs', 'spades']
    values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    
    print("\n正在載入卡片模板...")
    print(f"模板目錄: {CARDS_TEMPLATE_DIR}")
    
    if not os.path.exists(CARDS_TEMPLATE_DIR):
        print(f"錯誤: 模板目錄不存在: {CARDS_TEMPLATE_DIR}")
        return templates
    
    for suit in suits:
        for value in values:
            template_path = os.path.join(CARDS_TEMPLATE_DIR, f"{suit}_{value}.png")
            print(f"檢查模板: {template_path}")
            if os.path.exists(template_path):
                template = cv2.imread(template_path)
                if template is not None:
                    templates[(suit, value)] = template
                    print(f"成功載入: {suit}_{value}.png")
                else:
                    print(f"無法讀取圖片: {template_path}")
            else:
                print(f"模板不存在: {template_path}")
    
    print(f"共載入 {len(templates)} 個模板")
    return templates

def detect_cards(image, templates, threshold=0.6):
    """
    在圖片中檢測所有可見的撲克牌
    """
    detected_cards = []
    height, width = image.shape[:2]
    
    print(f"\n開始撲克牌檢測")
    print(f"原始圖片尺寸: {image.shape}")
    
    # 1. 圖像預處理
    # 調整亮度和對比度
    alpha = 1.2  # 對比度
    beta = 10    # 亮度
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 轉換為HSV色彩空間
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    
    # 保存預處理圖像
    timestamp = get_timestamp()
    cv2.imwrite(f"screenshots/adjusted_{timestamp}.png", adjusted)
    cv2.imwrite(f"screenshots/hsv_{timestamp}.png", hsv)
    
    # 2. 創建撲克牌顏色的遮罩
    # 白色區域的HSV範圍
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 保存遮罩
    cv2.imwrite(f"screenshots/white_mask_{timestamp}.png", white_mask)
    
    # 3. 形態學操作
    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(f"screenshots/morphed_mask_{timestamp}.png", white_mask)
    
    # 4. 找到輪廓
    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 過濾和分析輪廓
    min_card_area = 2000  # 增加最小面積
    max_card_area = 15000 # 調整最大面積
    
    debug_image = image.copy()
    valid_contours = []
    
    print(f"找到 {len(contours)} 個輪廓")
    
    # 首先收集所有有效的輪廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_card_area < area < max_card_area:
            # 獲取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 計算中心點
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            
            # 檢查坐標是否在圖片範圍內
            if 0 <= center_x < width and 0 <= center_y < height:
                valid_contours.append((contour, rect, box, (center_x, center_y), area))
    
    # 按面積排序輪廓（從大到小）
    valid_contours.sort(key=lambda x: x[4], reverse=True)
    
    # 處理每個有效的輪廓
    for i, (contour, rect, box, (center_x, center_y), area) in enumerate(valid_contours):
        # 在調試圖片上畫出輪廓和矩形
        cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 獲取矯正後的卡片圖像
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width < height:
            width, height = height, width
            
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                          [0, 0],
                          [width-1, 0],
                          [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(adjusted, M, (width, height))
        
        # 保存矯正後的圖像
        cv2.imwrite(f"screenshots/warped_{timestamp}_{i}.png", warped)
        
        # 分析顏色
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # 檢測紅色
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(warped_hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(warped_hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        # 檢測黑色
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(warped_hsv, lower_black, upper_black)
        
        # 保存顏色遮罩
        cv2.imwrite(f"screenshots/red_mask_{timestamp}_{i}.png", red_mask)
        cv2.imwrite(f"screenshots/black_mask_{timestamp}_{i}.png", black_mask)
        
        # 分析花色區域
        red_pixels = cv2.countNonZero(red_mask)
        black_pixels = cv2.countNonZero(black_mask)
        
        # 花色判斷邏輯
        suit = None
        if red_pixels > black_pixels * 0.3:  # 降低紅色判斷閾值
            if red_pixels > 100:
                # 分析紅色區域的形狀
                red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if red_contours:
                    largest_red = max(red_contours, key=cv2.contourArea)
                    moments = cv2.moments(largest_red)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # 計算輪廓的垂直對稱性
                        symmetry = calculate_symmetry(red_mask, cx)
                        suit = 'hearts' if symmetry > 0.8 else 'diamonds'
        else:
            if black_pixels > 100:
                # 分析黑色區域的形狀
                black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if black_contours:
                    largest_black = max(black_contours, key=cv2.contourArea)
                    moments = cv2.moments(largest_black)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # 計算輪廓的垂直對稱性
                        symmetry = calculate_symmetry(black_mask, cx)
                        suit = 'clubs' if symmetry > 0.8 else 'spades'
        
        if suit:
            print(f"輪廓 {i}: 面積={area:.1f}, 位置=({center_x}, {center_y})")
            print(f"紅色像素: {red_pixels}, 黑色像素: {black_pixels}")
            
            card = Card(
                suit=suit,
                value='Unknown',
                confidence=0.8,
                position=(center_x, center_y)
            )
            detected_cards.append(card)
    
    # 保存最終的調試圖片
    cv2.imwrite(f"screenshots/debug_{timestamp}.png", debug_image)
    
    return detected_cards

def calculate_symmetry(mask, center_x):
    """
    計算圖像相對於垂直軸的對稱性
    """
    left_half = mask[:, :center_x]
    right_half = mask[:, center_x:]
    
    # 確保兩半大小相同
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, -min_width:]
    right_half = right_half[:, :min_width]
    right_half = cv2.flip(right_half, 1)
    
    # 計算對稱性得分
    symmetric_pixels = np.sum(left_half == right_half)
    total_pixels = left_half.size
    
    return symmetric_pixels / total_pixels if total_pixels > 0 else 0

def filter_overlapping_cards(cards, distance_threshold=20):
    """
    過濾重疊的卡片檢測結果
    
    Args:
        cards: 檢測到的卡片列表
        distance_threshold: 判定為重疊的距離閾值
        
    Returns:
        list: 過濾後的卡片列表
    """
    filtered_cards = []
    cards.sort(key=lambda x: x.confidence, reverse=True)
    
    for card in cards:
        # 檢查是否與已保留的卡片重疊
        overlapping = False
        for kept_card in filtered_cards:
            if kept_card.position:
                distance = np.sqrt(
                    (card.position[0] - kept_card.position[0])**2 +
                    (card.position[1] - kept_card.position[1])**2
                )
                if distance < distance_threshold:
                    overlapping = True
                    break
        
        if not overlapping:
            filtered_cards.append(card)
    
    return filtered_cards

# 创建必要的目录
ensure_dir_exists(TEMPLATE_DIR)
ensure_dir_exists(SCREENSHOT_DIR)
ensure_dir_exists(CARDS_TEMPLATE_DIR)