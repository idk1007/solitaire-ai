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
    
    for suit in suits:
        for value in values:
            template_path = os.path.join(CARDS_TEMPLATE_DIR, f"{suit}_{value}.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path)
                if template is not None:
                    templates[(suit, value)] = template
    
    return templates

def detect_cards(image, templates, threshold=0.6):
    """
    在圖片中檢測所有可見的撲克牌
    """
    height, width = image.shape[:2]
    detected_cards = []
    
    print(f"\n開始撲克牌檢測")
    print(f"原始圖片尺寸: {image.shape}")
    
    # 1. 圖像預處理
    # 提高對比度和亮度
    adjusted = cv2.convertScaleAbs(image, alpha=1.3, beta=20)  # 降低對比度增強
    
    # 轉換到HSV空間
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    
    # 2. 創建撲克牌顏色的遮罩
    # 白色範圍（調整飽和度和亮度範圍）
    lower_white = np.array([0, 0, 180])  # 提高亮度下限
    upper_white = np.array([180, 40, 255])  # 提高飽和度上限
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 3. 形態學操作
    kernel = np.ones((3,3), np.uint8)  # 縮小kernel大小
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # 保存遮罩用於調試
    cv2.imwrite(f"screenshots/white_mask_{get_timestamp()}.png", white_mask)
    
    # 4. 找到輪廓
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 過濾和分析輪廓
    min_card_area = 9000  # 增大最小面積
    max_card_area = 11000  # 縮小最大面積
    valid_height = height * 0.8  # 設置有效高度範圍為圖片高度的80%
    
    debug_image = image.copy()
    valid_contours = []
    
    print(f"找到 {len(contours)} 個輪廓")
    
    # 收集有效輪廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_card_area <= area <= max_card_area:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 計算中心點
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            
            # 更嚴格的位置檢查
            if (50 <= center_x <= width-50 and 
                50 <= center_y <= valid_height and 
                center_y < height-100):  # 確保不在底部
                
                # 檢查寬高比
                w = max(rect[1])
                h = min(rect[1])
                if w > 0 and 1.4 <= w/h <= 1.7:  # 更嚴格的寬高比
                    valid_contours.append((contour, rect, box, (center_x, center_y), area))
    
    # 按面積排序
    valid_contours.sort(key=lambda x: x[4], reverse=True)
    
    timestamp = get_timestamp()
    
    # 處理每個有效的輪廓
    for i, (contour, rect, box, (center_x, center_y), area) in enumerate(valid_contours):
        # 在調試圖片上標記
        cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 獲取矯正後的卡片圖像
        width_rect = int(rect[1][0])
        height_rect = int(rect[1][1])
        if width_rect < height_rect:
            width_rect, height_rect = height_rect, width_rect
        
        # 透視變換
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height_rect-1],
                          [0, 0],
                          [width_rect-1, 0],
                          [width_rect-1, height_rect-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(adjusted, M, (width_rect, height_rect))
        
        # 保存矯正後的圖像
        cv2.imwrite(f"screenshots/warped_{timestamp}_{i}.png", warped)
        
        # 分析卡片
        card_info = analyze_card(warped, templates, (center_x, center_y), i, timestamp)
        
        if card_info:
            detected_cards.append(card_info)
            cv2.putText(debug_image, f"{card_info.suit}{card_info.value}", 
                       (center_x-20, center_y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存調試圖片
    cv2.imwrite(f"screenshots/debug_{timestamp}.png", debug_image)
    
    return detected_cards

def analyze_card(card_image, templates, position, index, timestamp):
    """
    分析單張卡片的花色和數值
    """
    height, width = card_image.shape[:2]
    
    # 1. 提取左上角區域
    roi_height = int(height * 0.3)  # 減小ROI區域
    roi_width = int(width * 0.3)
    roi = card_image[0:roi_height, 0:roi_width]
    
    # 保存ROI
    cv2.imwrite(f"screenshots/roi_{timestamp}_{index}.png", roi)
    
    # 2. 分析花色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 紅色檢測（調整範圍）
    lower_red1 = np.array([0, 150, 100])  # 提高飽和度下限
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2
    
    # 黑色檢測（調整範圍）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])  # 降低亮度上限
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 保存遮罩
    cv2.imwrite(f"screenshots/red_mask_{timestamp}_{index}.png", red_mask)
    cv2.imwrite(f"screenshots/black_mask_{timestamp}_{index}.png", black_mask)
    
    # 計算像素數量
    red_pixels = cv2.countNonZero(red_mask)
    black_pixels = cv2.countNonZero(black_mask)
    
    print(f"卡片 {index} - 紅色: {red_pixels}, 黑色: {black_pixels}")
    
    # 3. 確定花色（調整判定邏輯）
    suit = None
    if red_pixels > 1000 and red_pixels > black_pixels * 2:  # 更嚴格的紅色判定
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if red_contours:
            largest_red = max(red_contours, key=cv2.contourArea)
            red_area = cv2.contourArea(largest_red)
            red_perimeter = cv2.arcLength(largest_red, True)
            if red_perimeter > 0:
                circularity = 4 * np.pi * red_area / (red_perimeter * red_perimeter)
                suit = 'hearts' if circularity > 0.75 else 'diamonds'
    elif black_pixels > 1000 and black_pixels > red_pixels * 2:  # 更嚴格的黑色判定
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if black_contours:
            largest_black = max(black_contours, key=cv2.contourArea)
            black_area = cv2.contourArea(largest_black)
            black_perimeter = cv2.arcLength(largest_black, True)
            if black_perimeter > 0:
                circularity = 4 * np.pi * black_area / (black_perimeter * black_perimeter)
                suit = 'clubs' if circularity > 0.65 else 'spades'
    
    if not suit:
        return None
        
    # 4. 數值識別（改進模板匹配）
    # 轉換為灰度圖
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 自適應二值化
    thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # 保存處理後的ROI
    cv2.imwrite(f"screenshots/thresh_roi_{timestamp}_{index}.png", thresh_roi)
    
    # 模板匹配
    best_match = None
    best_score = -1
    
    # 多尺度匹配
    scales = [0.8, 0.9, 1.0, 1.1]
    
    for template_name, template in templates.items():
        if template_name.startswith(suit):
            template_value = template_name.split('_')[1]
            
            for scale in scales:
                # 調整模板大小
                template_resized = cv2.resize(template, None, fx=scale, fy=scale)
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
                _, template_thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
                
                # 確保大小合適
                if template_thresh.shape[0] > thresh_roi.shape[0] or \
                   template_thresh.shape[1] > thresh_roi.shape[1]:
                    continue
                
                try:
                    # 模板匹配
                    result = cv2.matchTemplate(thresh_roi, template_thresh, cv2.TM_CCOEFF_NORMED)
                    score = np.max(result)
                    
                    if score > best_score:
                        best_score = score
                        best_match = template_value
                except Exception as e:
                    continue
    
    # 提高匹配閾值
    value = best_match if best_score > 0.4 else 'Unknown'
    confidence = best_score if best_score > 0.4 else 0.8
    
    
    return Card(suit=suit, value=value, confidence=confidence, position=position)

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