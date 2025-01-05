import cv2
import numpy as np
from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
from adb_shell.auth.keygen import keygen
from PIL import Image
import io
import time
import os
import subprocess
import re
import sys
import platform

class PhoneController:
    def __init__(self, device_ip, port=5555):
        self.device_ip = device_ip
        self.port = port
        self.device = None
        self.connected = False
        self.device_id = f"{device_ip}:{port}"
        
        # 設置 ADB 路徑
        self.adb_path = self._get_adb_path()
        if not self.adb_path:
            print("Warning: adb command not found in PATH")
            print("Please install Android SDK Platform Tools and add it to PATH")
            print("Download from: https://developer.android.com/studio/releases/platform-tools")

        # 設置 ADB 金鑰路徑
        self.adb_key_path = os.path.expanduser('~/.android/adbkey')
        self.ensure_adb_key()

    def ensure_adb_key(self):
        """確保 ADB 金鑰存在"""
        try:
            # 創建 .android 目錄（如果不存在）
            key_dir = os.path.dirname(self.adb_key_path)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir)
                
            # 如果金鑰不存在，則生成新的金鑰對
            if not os.path.exists(self.adb_key_path):
                print("Generating new ADB key pair...")
                keygen(self.adb_key_path)
                
            # 檢查金鑰文件權限
            if platform.system() != 'Windows':
                # 在 Unix 系統上設置正確的文件權限
                os.chmod(self.adb_key_path, 0o600)
                if os.path.exists(self.adb_key_path + '.pub'):
                    os.chmod(self.adb_key_path + '.pub', 0o644)
                    
        except Exception as e:
            print(f"Warning: Failed to ensure ADB key: {str(e)}")
            print("You may need to manually set up ADB authentication")
        
    def _get_adb_path(self):
        """獲取 adb 可執行文件的路徑"""
        is_windows = platform.system() == 'Windows'
        adb_name = 'adb.exe' if is_windows else 'adb'
        nox_adb_name = 'nox_adb.exe' if is_windows else 'nox_adb'
        
        # 可能的 adb 路徑
        possible_paths = []
        
        if is_windows:
            # 優先檢查夜神模擬器路徑
            possible_paths.extend([
                r"C:\Program Files\Nox\bin\nox_adb.exe",
                r"C:\Program Files (x86)\Nox\bin\nox_adb.exe",
                # 備用的夜神路徑
                r"D:\Program Files\Nox\bin\nox_adb.exe",
                r"D:\Program Files (x86)\Nox\bin\nox_adb.exe",
                # 標準 ADB 路徑
                r"C:\Program Files\Android\android-sdk\platform-tools\adb.exe",
                r"C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe",
                os.path.expanduser("~/AppData/Local/Android/Sdk/platform-tools/adb.exe"),
                os.path.expanduser("~/scrcpy/adb.exe"),
            ])
        else:
            possible_paths.extend([
                "/usr/bin/adb",
                "/usr/local/bin/adb",
                os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            ])
        
        # 檢查環境變量
        for env_var in ['ANDROID_HOME', 'ANDROID_SDK_ROOT']:
            sdk_path = os.environ.get(env_var)
            if sdk_path:
                possible_paths.append(os.path.join(sdk_path, "platform-tools", adb_name))
        
        # 從 PATH 中查找
        if is_windows:
            for path in os.environ.get('PATH', '').split(';'):
                if path:
                    # 檢查 nox_adb
                    nox_path = os.path.join(path.strip('"'), nox_adb_name)
                    if os.path.exists(nox_path):
                        possible_paths.insert(0, nox_path)  # 將夜神 adb 放在最前面
                    # 檢查普通 adb
                    adb_path = os.path.join(path.strip('"'), adb_name)
                    possible_paths.append(adb_path)
        else:
            for path in os.environ.get('PATH', '').split(':'):
                if path:
                    possible_paths.append(os.path.join(path, adb_name))
        
        # 嘗試每個可能的路徑
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # 測試 adb 是否可用
                    result = subprocess.run([path, 'version'], 
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        check=True)
                    if 'Android Debug Bridge' in result.stdout:
                        print(f"Found ADB at: {path}")
                        return path
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
                
        # 最後嘗試直接使用 'nox_adb' 或 'adb' 命令
        try:
            if is_windows:
                result = subprocess.run(['nox_adb', 'version'], 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    check=True)
                if 'Android Debug Bridge' in result.stdout:
                    return 'nox_adb'
        except:
            try:
                result = subprocess.run(['adb', 'version'], 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True,
                                    check=True)
                if 'Android Debug Bridge' in result.stdout:
                    return 'adb'
            except:
                pass
                
        return None

    def _run_adb_command(self, command, **kwargs):
        """運行 ADB 命令，指定設備 ID"""
        if not self.adb_path:
            raise RuntimeError("ADB command not available. Please install Android SDK Platform Tools.")
            
        try:
            full_command = [self.adb_path, '-s', self.device_id] + command
            print(f"Running ADB command: {' '.join(full_command)}")
            result = subprocess.run(full_command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                                check=True,
                                **kwargs)
            return result
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            print(f"ADB command failed: {error_msg}")
            raise
    
    def _disconnect_all(self):
        """斷開所有設備連接"""
        if not self.adb_path:
            return
            
        try:
            subprocess.run([self.adb_path, 'disconnect'], check=True)
            print("Disconnected all devices")
        except subprocess.CalledProcessError:
            print("Warning: Failed to disconnect devices")
    
    def _get_connected_devices(self):
        """獲取已連接的設備列表"""
        if not self.adb_path:
            return []
            
        try:
            result = subprocess.run([self.adb_path, 'devices'], 
                                  capture_output=True, text=True, check=True)
            devices = []
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    match = re.match(r'(\S+)\s+device', line)
                    if match:
                        devices.append(match.group(1))
            return devices
        except subprocess.CalledProcessError:
            return []
    
    def connect(self):
        """連接到手機設備"""
        try:
            if not self.adb_path:
                raise RuntimeError("ADB not found. Please install Android SDK Platform Tools.")
                
            # 先斷開所有設備
            self._disconnect_all()
            time.sleep(1)
            
            # 啟動 ADB 服務器
            try:
                subprocess.run([self.adb_path, 'start-server'], check=True)
                print("ADB server started")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not start ADB server: {e}")
            
            # 檢查當前連接的設備
            devices = self._get_connected_devices()
            if self.device_id in devices:
                print(f"Device {self.device_id} already connected")
            else:
                # 嘗試連接設備
                try:
                    # 修改這裡，使用 stdout 和 stderr 替代 capture_output
                    result = subprocess.run(
                        [self.adb_path, 'connect', self.device_id],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    print(f"ADB connected to {self.device_id}")
                    time.sleep(1)  # 等待連接建立
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not connect using adb command: {e.stderr.decode() if e.stderr else str(e)}")
            
            # 創建 ADB 連接
            self.device = AdbDeviceTcp(self.device_ip, self.port, default_transport_timeout_s=9.)
            
            # 載入 ADB 金鑰
            signer = PythonRSASigner.FromRSAKeyPath(self.adb_key_path)
            
            # 連接設備
            self.device.connect(rsa_keys=[signer], auth_timeout_s=5)
            
            # 測試連接
            result = self.device.shell('echo "Testing connection"')
            if result.strip() == "Testing connection":
                print(f"Successfully connected to {self.device_ip}")
                self.connected = True
                return True
            else:
                print("Connection test failed")
                return False
                
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            self.connected = False
            return False

    def _get_connected_devices(self):
        """獲取已連接的設備列表"""
        if not self.adb_path:
            return []
            
        try:
            # 修改這裡，使用 stdout 和 stderr 替代 capture_output
            result = subprocess.run(
                [self.adb_path, 'devices'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            devices = []
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    match = re.match(r'(\S+)\s+device', line)
                    if match:
                        devices.append(match.group(1))
            return devices
        except subprocess.CalledProcessError:
            return []
            
    def disconnect(self):
        """斷開與設備的連接"""
        if self.device:
            try:
                self.device.close()
                print("Device disconnected")
                if self.adb_path:
                    subprocess.run([self.adb_path, 'disconnect', self.device_id])
            except:
                pass
        self.connected = False
    
    def get_screen_capture(self):
        """獲取螢幕截圖"""
        if not self.connected:
            print("Device not connected")
            return None
            
        try:
            # 使用臨時文件來保存截圖
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(script_dir, 'temp')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            temp_file = os.path.join(temp_dir, f'screenshot_{self.device_ip.replace(".", "_")}.png')
            remote_file = '/sdcard/screenshot.png'
            
            # 使用 adb 命令獲取截圖
            try:
                if self.adb_path:
                    print("Capturing screenshot using ADB command...")
                    self._run_adb_command(['shell', 'screencap', '-p', remote_file])
                    self._run_adb_command(['pull', remote_file, temp_file])
                    self._run_adb_command(['shell', 'rm', remote_file])
                    
                    # 讀取圖像
                    image = cv2.imread(temp_file)
                    if image is None:
                        raise ValueError("Failed to read captured image")
                    
                    # 刪除臨時文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                    return image
                    
            except Exception as e:
                print(f"Failed to capture screen using adb command: {str(e)}")
                print("Trying adb_shell method...")
                
                # 如果 adb 命令失敗，嘗試使用 adb_shell
                result = self.device.shell('screencap -p', decode=False)
                if result:
                    # 修正換行符
                    result = result.replace(b'\r\n', b'\n')
                    # 將輸出轉換為圖像
                    with open(temp_file, 'wb') as f:
                        f.write(result)
                    image = cv2.imread(temp_file)
                    if image is None:
                        raise ValueError("Failed to read captured image")
                    
                    # 刪除臨時文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                    return image
                    
        except Exception as e:
            print(f"Screenshot failed: {str(e)}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return None
    
    def tap(self, x, y):
        """點擊螢幕上的指定位置"""
        if not self.connected:
            return False
            
        try:
            # 先嘗試使用 adb 命令
            try:
                if self.adb_path:
                    self._run_adb_command(['shell', 'input', 'tap', str(int(x)), str(int(y))])
                    return True
            except:
                # 如果失敗，使用 adb_shell
                self.device.shell(f'input tap {int(x)} {int(y)}')
                return True
        except Exception as e:
            print(f"Tap failed: {str(e)}")
            return False
    
    def swipe(self, start_x, start_y, end_x, end_y, duration=300):
        """滑動操作"""
        if not self.connected:
            return False
            
        try:
            # 先嘗試使用 adb 命令
            try:
                if self.adb_path:
                    self._run_adb_command([
                        'shell', 'input', 'swipe',
                        str(int(start_x)), str(int(start_y)),
                        str(int(end_x)), str(int(end_y)),
                        str(duration)
                    ])
                    return True
            except:
                # 如果失敗，使用 adb_shell
                self.device.shell(
                    f'input swipe {int(start_x)} {int(start_y)} {int(end_x)} {int(end_y)} {duration}'
                )
                return True
        except Exception as e:
            print(f"Swipe failed: {str(e)}")
            return False