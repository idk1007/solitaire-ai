import cv2
import numpy as np

def detect_cards(image):
    """
    檢測圖像中的撲克牌
    
    Args:
        image: OpenCV 圖像
    
    Returns:
        tuple: (檢測結果, 卡牌列表)
    """
    try:
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自適應二值化
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 尋找輪廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        cards = []
        min_card_area = 1000  # 最小卡牌面積
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_card_area:
                continue
                
            # 獲取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 計算中心點
            center = np.mean(box, axis=0)
            
            # 提取卡牌圖像
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                              [0, 0],
                              [width-1, 0],
                              [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            card_img = cv2.warpPerspective(image, M, (width, height))
            
            cards.append({
                'image': card_img,
                'position': (int(center[0]), int(center[1])),
                'box': box
            })
            
        return True, cards
        
    except Exception as e:
        print(f"Card detection error: {e}")
        return False, []

def recognize_card(card_image):
    """
    識別卡牌的花色和數字
    
    Args:
        card_image: 卡牌圖像
    
    Returns:
        tuple: (花色, 數字) 或 None
    """
    try:
        # 轉換為灰度圖
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # 提取左上角區域（通常包含花色和數字）
        height, width = gray.shape
        roi = gray[0:int(height*0.2), 0:int(width*0.2)]
        
        # TODO: 實現具體的卡牌識別邏輯
        # 這裡需要使用OCR或其他圖像識別方法
        # 暫時返回None，表示無法識別
        return None
        
    except Exception as e:
        print(f"Card recognition error: {e}")
        return None

def draw_debug_info(image, cards):
    """
    在圖像上繪製調試信息
    
    Args:
        image: 原始圖像
        cards: 檢測到的卡牌列表
    
    Returns:
        標註後的圖像
    """
    debug_image = image.copy()
    
    for card in cards:
        # 繪製檢測到的卡牌輪廓
        cv2.drawContours(debug_image, [card['box']], 0, (0, 255, 0), 2)
        
        # 標記中心點
        center = card['position']
        cv2.circle(debug_image, center, 5, (0, 0, 255), -1)
        
    return debug_image
