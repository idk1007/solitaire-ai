from phone_controller import PhoneController

def main():
    # 創建控制器實例，使用夜神模擬器的默認地址和端口
    controller = PhoneController("127.0.0.1", 62001)
    
    # 嘗試連接到模擬器
    if controller.connect():
        print("成功連接到夜神模擬器")
        
        try:
            # 測試截圖
            image = controller.get_screen_capture()
            if image is not None:
                print("成功獲取截圖")
            
            # 測試點擊 (例如點擊座標 100, 100)
            if controller.tap(100, 100):
                print("成功執行點擊操作")
            
            # 測試滑動 (例如從 (100, 300) 滑動到 (100, 100))
            if controller.swipe(100, 300, 100, 100):
                print("成功執行滑動操作")
                
        except Exception as e:
            print(f"操作出錯: {str(e)}")
            
        finally:
            # 最後斷開連接
            controller.disconnect()
            print("已斷開連接")
    else:
        print("連接失敗")

if __name__ == "__main__":
    main()