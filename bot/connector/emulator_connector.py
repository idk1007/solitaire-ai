import os
import subprocess
import time
from utils.utils import find_template
from config.config import ADB_PATH, DEFAULT_PORT

class EmulatorConnector:
    def __init__(self):
        self.adb_path = ADB_PATH
        self.port = DEFAULT_PORT
        self.connected = False
        self.device_id = None  # 添加設備ID屬性

    def _run_command(self, cmd):
        """
        运行命令并返回输出
        """
        try:
            # 如果有指定設備ID，在命令中添加 -s 參數
            if self.device_id and not cmd.endswith('devices') and 'connect' not in cmd:
                cmd = f"{self.adb_path} -s {self.device_id} {cmd.replace(self.adb_path, '').strip()}"
            
            print(f"執行命令: {cmd}")
            
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True
                )
            else:  # Linux/Mac
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
            
            stdout, stderr = process.communicate()
            
            if stdout:
                print(f"命令輸出: {stdout.strip()}")
            if stderr:
                print(f"錯誤輸出: {stderr.strip()}")
            print(f"返回代碼: {process.returncode}")
            
            return stdout, stderr, process.returncode
            
        except Exception as e:
            print(f"执行命令时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, str(e), -1

    def get_device_list(self):
        """
        獲取所有連接的設備列表
        """
        stdout, _, _ = self._run_command(f"{self.adb_path} devices")
        devices = []
        if stdout:
            lines = stdout.strip().split('\n')
            for line in lines[1:]:  # 跳過第一行（標題行）
                if line.strip():
                    device_id = line.split('\t')[0]
                    devices.append(device_id)
        return devices

    def connect(self):
        """
        连接到模拟器
        """
        # 先斷開所有連接
        self._run_command(f"{self.adb_path} disconnect")
        print("正在连接模拟器...")
        try:
            # 检查ADB路径是否存在
            if not os.path.exists(self.adb_path):
                print(f"错误: 找不到ADB工具于 {self.adb_path}")
                return False

            # 启动ADB服务器
            self._run_command(f"{self.adb_path} start-server")
            
            # 连接到模拟器
            stdout, stderr, returncode = self._run_command(
                f"{self.adb_path} connect 127.0.0.1:{self.port}"
            )
            
            if stdout and 'connected' in stdout.lower():
                # 獲取設備列表
                devices = self.get_device_list()
                if devices:
                    # 使用第一個設備
                    self.device_id = devices[0]
                    print(f"選擇設備: {self.device_id}")
                    self.connected = True
                    print("成功连接到模拟器")
                    return True
                else:
                    print("未找到可用設備")
                    return False
            else:
                print(f"连接失败: {stdout}")
                if stderr:
                    print(f"错误信息: {stderr}")
                return False

        except Exception as e:
            print(f"连接过程发生错误: {e}")
            return False

    def screenshot(self, filename):
        """
        截取模拟器屏幕截图
        """
        try:
            if not self.connected or not self.device_id:
                print("未连接到模拟器或未指定設備")
                return False

            print("正在执行截图操作...")
            
            # 確保目標目錄存在
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # 先将截图保存到模拟器
            print("步驟 1: 在模擬器中截圖")
            cmd = f"shell screencap -p /sdcard/screenshot.png"
            stdout, stderr, returncode = self._run_command(cmd)
            if returncode != 0:
                print(f"模擬器截圖失敗: {stderr}")
                return False
                
            # 将截图拉到电脑
            print("步驟 2: 將截圖傳輸到電腦")
            cmd = f"pull /sdcard/screenshot.png \"{filename}\""
            stdout, stderr, returncode = self._run_command(cmd)
            if returncode != 0:
                print(f"傳輸截圖失敗: {stderr}")
                return False
                
            # 删除模拟器中的截图
            print("步驟 3: 清理模擬器中的暫存截圖")
            cmd = f"shell rm /sdcard/screenshot.png"
            self._run_command(cmd)
            
            # 驗證文件是否成功創建
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"截圖成功: {filename} (大小: {file_size} bytes)")
                return True
            else:
                print(f"截圖文件未創建: {filename}")
                return False
                
        except Exception as e:
            print(f"截图过程发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def tap(self, x, y):
        """
        模拟点击屏幕
        """
        try:
            if not self.connected:
                print("未连接到模拟器")
                return False

            self._run_command(f"{self.adb_path} shell input tap {x} {y}")
            return True
        except Exception as e:
            print(f"点击操作时发生错误: {e}")
            return False

    def swipe(self, x1, y1, x2, y2, duration=500):
        """
        模拟滑动屏幕
        duration: 滑动持续时间(毫秒)
        """
        try:
            if not self.connected:
                print("未连接到模拟器")
                return False

            self._run_command(
                f"{self.adb_path} shell input swipe {x1} {y1} {x2} {y2} {duration}"
            )
            return True
        except Exception as e:
            print(f"滑动操作时发生错误: {e}")
            return False

    def check_app_running(self, package_name):
        """
        检查指定应用是否正在运行
        """
        try:
            stdout, _, _ = self._run_command(f"{self.adb_path} shell pidof {package_name}")
            return bool(stdout and stdout.strip())
        except Exception as e:
            print(f"检查应用运行状态时发生错误: {e}")
            return False

    def start_app(self, package_name, activity_name=None):
        """
        启动指定应用
        """
        try:
            if activity_name:
                cmd = f"{self.adb_path} shell am start -n {package_name}/{activity_name}"
            else:
                cmd = f"{self.adb_path} shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
            
            self._run_command(cmd)
            return True
        except Exception as e:
            print(f"启动应用时发生错误: {e}")
            return False

    def stop_app(self, package_name):
        """
        停止指定应用
        """
        try:
            self._run_command(f"{self.adb_path} shell am force-stop {package_name}")
            return True
        except Exception as e:
            print(f"停止应用时发生错误: {e}")
            return False