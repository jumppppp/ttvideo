import sys
import time
import cv2
import numpy as np
import psutil
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import pyautogui
from ui import Ui_MainWindow
import os
import win32gui
import win32process
import dxcam
import threading
import subprocess
import traceback
import GPUtil
# 录制时长
all_time = 0

class PreviewThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    performance_updated = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, blur_process_names):
        super().__init__(parent=None)
        self.blur_process_names  = [kw.lower() for kw in blur_process_names]  # 存储小写关键字
        self._stop_flag = False
        self.camera = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.screen_width, self.screen_height = pyautogui.size()
        self.retry_count = 0

    def init_camera(self):
        """初始化DXCam截图相机"""
        try:
            # 使用实际屏幕分辨率
            self.camera = dxcam.create(
                output_idx=0, 
                output_color="BGR",
                max_buffer_len=64  # 增加缓冲区大小
            )
            if self.camera:
                self.camera.start(target_fps=30, video_mode=True)
                return True
            return False
        except Exception as e:
            error_msg = f"初始化DXCam失败: {str(e)}"
            self.error_occurred.emit(error_msg)
            traceback.print_exc()
            return False

    def run(self):
        # 多次尝试初始化相机
        while not self._stop_flag and not self.init_camera() and self.retry_count < 5:
            self.retry_count += 1
            self.error_occurred.emit(f"截图设备初始化失败，正在重试 ({self.retry_count}/5)...")
            time.sleep(1)
            
        if not self.camera:
            self.error_occurred.emit("错误: 无法初始化截图设备，预览不可用")
            return

        self.performance_updated.emit("预览已启动")
        
        while not self._stop_flag:
            try:
                start_time = time.time()
                
                # 高性能截图
                frame = self.camera.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 应用模糊效果
                if self.blur_process_names:
                    frame = self.blur_windows(frame)
                
                # 确保图像格式正确
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # 发送预览帧
                self.frame_ready.emit(frame)
                
                # 计算FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 30:
                    fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.performance_updated.emit(f"预览FPS: {fps:.1f}")
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, 0.08 - elapsed)  # 目标30FPS预览
                time.sleep(sleep_time)
                
            except Exception as e:
                error_msg = f"预览处理出错: {str(e)}"
                self.error_occurred.emit(error_msg)
                traceback.print_exc()
                time.sleep(0.1)

    def fast_blur(self, region):
        """将窗口区域变为全白色"""
        try:
            if region.size == 0:
                return region
                
            # 获取区域尺寸
            h, w = region.shape[:2]
            if w > 0 and h > 0:
                # 创建全白色的图像区域
                white_region = np.full((h, w, 3), 255, dtype=np.uint8)
                return white_region
            return region
        except Exception as e:
            self.error_occurred.emit(f"白色处理出错: {str(e)}")
            return region

    def blur_windows(self, frame):
        """对指定进程名称的窗口进行模糊处理"""
        try:
            processed_frame = frame.copy()
            
            def enum_windows_callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        # 通过PID获取进程名称（使用psutil）
                        process = psutil.Process(pid)
                        process_name = process.name().lower()  # 转为小写方便匹配
                        
                        # 检查进程名称是否包含目标关键字（如"wechat"、"qq"）
                        if any(keyword in process_name for keyword in self.blur_process_names):
                            # 获取窗口坐标（保持原逻辑）
                            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                            width = right - left
                            height = bottom - top
                            
                            if (width > 10 and height > 10 and 
                                left >= 0 and top >= 0 and
                                left + width <= self.screen_width and
                                top + height <= self.screen_height):
                                
                                window_region = processed_frame[top:top+height, left:left+width]
                                if window_region.size > 0:
                                    blurred_region = self.fast_blur(window_region)
                                    processed_frame[top:top+height, left:left+width] = blurred_region
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # 进程已结束或无权限访问时跳过
                        pass
                    except Exception:
                        pass
                return True
            
            win32gui.EnumWindows(enum_windows_callback, None)
            return processed_frame
        except Exception as e:
            self.error_occurred.emit(f"窗口模糊处理出错: {str(e)}")
            return frame

    def stop(self):
        self._stop_flag = True
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
        self.wait(2000)  # 最多等待2秒

class RecordingThread(QtCore.QThread):
    frame_written = QtCore.pyqtSignal(int)
    performance_updated = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    
    def __init__(self, recorder):
        super().__init__(parent=None)
        self.recorder = recorder
        self._stop_flag = False
        self._pause_flag = False
        self.frame_count = 0
        self.camera = None
        self.writer = None
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.screen_width, self.screen_height = pyautogui.size()

    def init_camera(self):
        """初始化DXCam截图相机"""
        try:
            self.camera = dxcam.create(
                output_idx=0, 
                output_color="BGR",
                max_buffer_len=128  # 更大的缓冲区
            )
            if self.camera:
                self.camera.start(target_fps=self.recorder.fps, video_mode=True)
                return True
            return False
        except Exception as e:
            error_msg = f"录制截图设备初始化失败: {str(e)}"
            self.error_occurred.emit(error_msg)
            traceback.print_exc()
            return False
    def check_hardware_support(self):
        """检查系统硬件编码支持"""
        support = {
            "NVIDIA": False,
            "AMD": False,
            "Intel": False
        }
        
        try:
            # 检查 NVIDIA
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                support["NVIDIA"] = True
            pynvml.nvmlShutdown()
        except:
            pass
        
        try:
            # 检查 AMD (需要pyadl)
            from pyadl import ADLManager
            ADLManager.getInstance()
            support["AMD"] = True
        except:
            pass
        
        # Intel 通常通过 QSV
        try:
            from winreg import OpenKey, HKEY_LOCAL_MACHINE, QueryValueEx
            key = OpenKey(HKEY_LOCAL_MACHINE, r"SOFTWARE\Intel\MediaSDK")
            support["Intel"] = True
        except:
            pass
        
        return support

    def init_writer(self, output_path):
        """初始化视频写入器，优先使用硬件编码"""
        try:
            # 添加硬件编码器检测信息
            gpu_info = "未知"
            try:
                
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = f"{gpus[0].name} (VRAM: {gpus[0].memoryTotal}MB)"
            except:
                pass
            
            self.performance_updated.emit(f"检测到GPU: {gpu_info}")
            
            # 检查硬件支持
            hw_support = self.check_hardware_support()
            support_info = ", ".join([f"{k}: {'是' if v else '否'}" for k, v in hw_support.items()])
            self.performance_updated.emit(f"硬件编码支持: {support_info}")
            
            # 扩展编码器列表，添加更多选项
            codecs = [
                ('nvenc', 'NVIDIA H.264'),        # NVIDIA
                ('h264_nvenc', 'NVIDIA H.264'),   # NVIDIA 备选
                ('h264_amf', 'AMD H.264'),        # AMD
                ('h264_qsv', 'Intel H.264'),      # Intel
                ('avc1', 'Apple H.264'),          # macOS
                ('X264', 'Software H.264'),       # x264软件编码器
                ('mp4v', 'Software MPEG-4')       # 默认软件编码
            ]
            
            # 尝试所有编码器
            errors = []
            for codec, name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        self.recorder.fps,
                        self.recorder.resolution
                    )
                    if writer.isOpened():
                        # 尝试设置编码器参数（如果支持）
                        try:
                            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)  # 设置质量
                            writer.set(cv2.VIDEOWRITER_PROP_NV_PRESET, "fast")  # NVIDIA 特定设置
                            self.performance_updated.emit("使用NVIDIA H.264编码器 (快速预设)")
                        except:
                            self.performance_updated.emit("使用NVIDIA H.264编码器")
                        return writer
                except Exception as e:
                    # 关键修改：记录具体错误信息
                    errors.append(f"{name}({codec}): {str(e)}")
                    self.error_occurred.emit(f"尝试编码器 {name} 失败: {str(e)}")  # 输出到UI日志
                    continue
            
            # 记录所有错误
            self.error_occurred.emit(f"硬件编码器初始化失败:\n" + "\n".join(errors))
            
            # 尝试使用默认编码器
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.recorder.fps,
                self.recorder.resolution
            )
            if writer.isOpened():
                self.performance_updated.emit("使用默认软件编码器")
                return writer
                
            self.error_occurred.emit("错误: 无法创建视频文件，所有编码器均失败")
            return None
        except Exception as e:
            self.error_occurred.emit(f"视频写入器初始化失败: {str(e)}")
            return None

    def run(self):
        global all_time
        current_time = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"./output/recording_{current_time}.mp4"
        self.recorder.output_path = output_path

        # 检查分辨率有效性
        if self.recorder.resolution[0] <= 0 or self.recorder.resolution[1] <= 0:
            self.recorder.resolution = (1920, 1080)

        # 初始化截图设备
        if not self.init_camera():
            self.error_occurred.emit("错误: 无法初始化截图设备")
            return

        # 初始化视频写入器
        self.writer = self.init_writer(output_path)
        if not self.writer or not self.writer.isOpened():
            self.error_occurred.emit("错误: 无法创建视频文件")
            return

        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.frame_count = 0
        
        # 计算目标帧间隔
        target_interval = 1.0 / self.recorder.fps
        next_frame_time = time.time()
        
        # 主录制循环
        while not self._stop_flag and self.recorder.recording:
            try:
                if not self._pause_flag:
                    # 获取当前时间
                    current_time = time.time()
                    
                    # 高性能截图
                    frame = self.camera.get_latest_frame()
                    
                    if frame is None:
                        # 等待下一帧
                        sleep_time = next_frame_time - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue
                    
                    # 应用模糊效果
                    if self.recorder.blur_process_names:
                        frame = self.recorder.preview_thread.blur_windows(frame)
                    
                    # 调整分辨率
                    if frame.shape[:2] != self.recorder.resolution[::-1]:
                        frame = cv2.resize(frame, self.recorder.resolution)
                    
                    # 写入视频
                    self.writer.write(frame)
                    self.frame_count += 1
                    
                    # 性能监控
                    if current_time - self.last_log_time >= 5.0:
                        elapsed = current_time - self.start_time
                        actual_fps = self.frame_count / elapsed
                        self.performance_updated.emit(
                            f"录制FPS: {actual_fps:.1f}/{self.recorder.fps} | "
                            f"帧数: {self.frame_count} | "
                            f"文件: {os.path.basename(output_path)}"
                        )
                        self.last_log_time = current_time
                    
                    # 精确控制帧率
                    next_frame_time += target_interval
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # 帧处理超时，调整下一帧时间
                        next_frame_time = time.time()
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.error_occurred.emit(f"录制过程中出错: {str(e)}")
                time.sleep(0.1)
        
        # 释放资源
        try:
            if self.writer:
                self.writer.release()
            if self.camera:
                self.camera.stop()
        except:
            pass
        
        # 最终报告
        total_time = all_time
        actual_fps = self.frame_count / total_time if total_time > 0 else 0
        theoretical_duration = self.frame_count / self.recorder.fps
        msg = (f"录制完成: {self.frame_count}帧 | "
            f"耗时: {total_time:.1f}秒 | "
            f"平均FPS: {actual_fps:.1f} | "
            f"理论时长: {theoretical_duration:.1f}秒")
        self.performance_updated.emit(msg)

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def stop(self):
        self._stop_flag = True
        self.wait(3000)  # 最多等待3秒
        
class TimeDisplayThread(QtCore.QThread):

    time_updated = QtCore.pyqtSignal(str)

    def __init__(self, recorder):
        super().__init__(parent=None)
        self.recorder = recorder
        self._stop_flag = False
        self.last_recorded_time = 0

    def run(self):
        global all_time
        while not self._stop_flag:
            try:
                if self.recorder.recording:
                    if not self.recorder.paused:
                        # 计算已录制时间（减去暂停时间）
                        elapsed_time = time.time() - self.recorder.start_time - self.recorder.total_pause_time
                        self.last_recorded_time = elapsed_time
                        self.time_updated.emit(f"录制中: {int(elapsed_time)}秒")
                        
                    else:
                        # 暂停状态
                        self.time_updated.emit(f"已暂停: {int(self.last_recorded_time)}秒")
                    all_time = self.last_recorded_time
                else:
                    # 未录制状态
                    self.time_updated.emit("就绪")
                
                self.msleep(200)  # 每200ms更新一次
            except Exception:
                pass

    def stop(self):
        self._stop_flag = True
        self.wait()
class ScreenRecorder(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化变量
        self.recording = False
        self.paused = False
        self.start_time = 0
        self.total_pause_time = 0
        self.pause_start_time = 0
        self.fps = 24  # 默认帧率设为30
        self.resolution = (1920, 1080)
        self.blur_process_names = []  # 替换原来的blur_pids
        self.output_path = ""
        self.recording_thread = None
        # self.screen_width, self.screen_height = pyautogui.size()
        # 添加共享的 DXCamera 实例
        # self.shared_camera = None
        
        # 在创建预览和录制线程时传递共享实例

        # 设置预览标签背景色
        self.ui.label.setStyleSheet("background-color: #2D2D30;")
        self.ui.label.setText("预览初始化中...")
        self.ui.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # 确保output目录存在
        if not os.path.exists('./output'):
            os.makedirs('./output')

        # 连接信号与槽
        self.ui.pushButton.clicked.connect(self.start_recording)
        self.ui.pushButton_2.clicked.connect(self.toggle_pause_resume)
        self.ui.pushButton_3.clicked.connect(self.stop_recording)
        self.ui.comboBox.currentTextChanged.connect(self.set_resolution)
        self.ui.comboBox_2.currentTextChanged.connect(self.set_fps)
        self.ui.pushButton_4.clicked.connect(self.add_blur_process)
        self.ui.pushButton_5.clicked.connect(self.open_output_folder)

        # 初始化分辨率选项
        resolutions = [
            # f"{self.screen_width}x{self.screen_height}",
            "1920x1080", 
            "1280x720", 
            "800x600"
        ]
        self.ui.comboBox.addItems(resolutions)
        self.ui.comboBox_2.addItems(["15", "24", "30"])
        self.ui.comboBox.setCurrentText(f"1920x1080")
        self.ui.comboBox_2.setCurrentText("24")

        # 初始按钮状态
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)

        # 初始状态
        self.update_status("就绪，等待开始录制")
        
        # 初始化预览线程（稍后启动）
        # self.preview_thread = PreviewThread(self.blur_pids)
        self.preview_thread = PreviewThread(self.blur_process_names)
        self.preview_thread.frame_ready.connect(self.update_preview)
        self.preview_thread.performance_updated.connect(self.update_status)
        self.preview_thread.error_occurred.connect(self.update_status)
        
        # 初始化时间显示线程
        self.time_display_thread = TimeDisplayThread(self)
        self.time_display_thread.time_updated.connect(self.update_time_label)
        self.time_display_thread.start()
        

    def showEvent(self, event):
        """窗口显示后启动预览"""
        super().showEvent(event)
        # 延迟启动预览线程，确保UI已完全初始化
        QtCore.QTimer.singleShot(500, self.start_preview)

    def start_preview(self):
        """启动预览线程"""
        if not self.preview_thread.isRunning():
            self.preview_thread.start()
            self.update_status("正在启动预览...")

    def update_preview(self, frame):
        """更新预览画面"""
        try:
            # 确保图像格式正确
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            # 检查图像尺寸
            if frame.size == 0:
                return
                
            # 调整预览大小
            label_width = self.ui.label.width()
            label_height = self.ui.label.height()
            
            if label_width <= 0 or label_height <= 0:
                return
                
            # 保持宽高比缩放
            h, w, _ = frame.shape
            aspect_ratio = w / h
            target_height = label_height
            target_width = int(target_height * aspect_ratio)
            
            if target_width > label_width:
                target_width = label_width
                target_height = int(target_width / aspect_ratio)
                
            # 调整大小
            preview_frame = cv2.resize(frame, (target_width, target_height))
            
            # 转换为Qt图像格式 (BGR -> RGB)
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = preview_frame.shape
            bytes_per_line = 3 * width
            
            # 创建QImage
            q_img = QImage(
                preview_frame.data, 
                width, 
                height, 
                bytes_per_line, 
                QImage.Format.Format_RGB888
            )
            
            # 创建QPixmap并设置到标签
            pixmap = QPixmap.fromImage(q_img)
            self.ui.label.setPixmap(pixmap)
            
        except Exception as e:
            error_msg = f"更新预览出错: {str(e)}"
            self.update_status(error_msg)
            # 设置占位符
            self.ui.label.setText("预览错误")
            self.ui.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    def update_time_label(self, time_text):
        """更新时间显示标签的内容"""
        self.ui.label_2.setText(time_text)
    def update_status(self, message):
        """更新状态栏"""
        timestamp = time.strftime('%H:%M:%S')
        self.ui.plainTextEdit_2.appendPlainText(f"[{timestamp}] {message}")
        # 自动滚动到底部
        self.ui.plainTextEdit_2.verticalScrollBar().setValue(
            self.ui.plainTextEdit_2.verticalScrollBar().maximum()
        )

    def start_recording(self):
        """开始录制"""
        if not self.recording:
            self.recording = True
            self.paused = False
            self.start_time = time.time()
            self.total_pause_time = 0
            self.pause_start_time = 0
            
            # 创建录制线程
            self.recording_thread = RecordingThread(self)
            self.recording_thread.performance_updated.connect(self.update_status)
            self.recording_thread.error_occurred.connect(self.update_status)
            self.recording_thread.start()
            
            # 更新UI
            self.update_status(f"开始录制: {self.fps}FPS {self.resolution[0]}x{self.resolution[1]}")
            self.ui.pushButton.setEnabled(False)
            self.ui.pushButton_2.setEnabled(True)
            self.ui.pushButton_3.setEnabled(True)
            self.ui.pushButton_2.setText("暂停")

    def toggle_pause_resume(self):
        """暂停/继续录制"""
        if self.recording:
            if self.paused:
                # 继续录制
                self.paused = False
                if self.recording_thread:
                    self.recording_thread.resume()
                self.ui.pushButton_2.setText("暂停")
                self.total_pause_time += time.time() - self.pause_start_time
                self.update_status("继续录制")

            else:
                # 暂停录制
                self.paused = True
                if self.recording_thread:
                    self.recording_thread.pause()
                self.ui.pushButton_2.setText("继续")
                self.pause_start_time = time.time()
                self.update_status("录制已暂停")

    def stop_recording(self):
        """停止录制"""
        if self.recording:
            self.recording = False
            self.paused = False
            
            if self.recording_thread:
                self.recording_thread.stop()
                self.recording_thread = None
            
            # 计算总录制时间
            total_time = time.time() - self.start_time - self.total_pause_time
            self.update_status(f"录制已停止 | 时长: {total_time:.1f}秒")
            
            # 更新UI
            self.ui.pushButton.setEnabled(True)
            self.ui.pushButton_2.setEnabled(False)
            self.ui.pushButton_3.setEnabled(False)
            self.ui.pushButton_2.setText("暂停/继续")
            
            # 不要停止预览线程，让它继续运行

    def set_resolution(self, text):
        """设置分辨率"""
        try:
            width, height = map(int, text.split('x'))
            self.resolution = (width, height)
            self.update_status(f"分辨率设置为: {width}x{height}")
        except ValueError:
            self.update_status("错误: 无效的分辨率格式")

    def set_fps(self, text):
        """设置帧率"""
        try:
            self.fps = int(text)
            self.update_status(f"帧率设置为: {self.fps} FPS")
        except ValueError:
            self.update_status("错误: 无效的帧率格式")

    def add_blur_process(self):  # 替换原来的add_blur_pid方法
        """添加需要模糊的进程名称关键字"""
        process_text = self.ui.textEdit.toPlainText()  # 假设UI输入框用于输入进程名称
        keywords = [kw.strip().lower() for kw in process_text.split('\n') if kw.strip()]
        added = []
        
        for kw in keywords:
            if kw not in self.blur_process_names:
                self.blur_process_names.append(kw)
                added.append(kw)
        
        if added:
            self.preview_thread.blur_process_names = self.blur_process_names  # 同步到预览线程
            self.update_status(f"添加模糊进程关键字: {', '.join(added)}")
        else:
            self.update_status("错误: 未添加有效进程名称关键字")

    def open_output_folder(self):
        """打开输出文件夹"""
        output_dir = os.path.abspath("./output")
        try:
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
            self.update_status(f"已打开输出目录: {output_dir}")
        except Exception as e:
            self.update_status(f"无法打开文件夹: {str(e)}")

    def closeEvent(self, event):
        """关闭应用程序"""
        self.update_status("正在关闭应用程序...")
        
        # 停止所有线程
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.stop()
        
        if self.recording and self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
        
        # 停止时间显示线程
        if self.time_display_thread and self.time_display_thread.isRunning():
            self.time_display_thread.stop()
        
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建深色主题
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(45, 45, 48))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 48))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 48))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(0, 122, 204))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    # 创建主窗口
    recorder = ScreenRecorder()
    recorder.setWindowTitle("TTvideo录制软件")
    recorder.resize(1000, 700)
    
    # 应用图标
    if hasattr(QtGui, "QIcon"):
        app_icon = QtGui.QIcon()
        recorder.setWindowIcon(app_icon)
    
    recorder.show()
    
    sys.exit(app.exec())