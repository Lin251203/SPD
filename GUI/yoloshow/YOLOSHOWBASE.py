import random

from utils import glo
from utils.logger import LoggerUtils
import re
import socket
from urllib.parse import urlparse
import torch
import json
import os
import shutil
import cv2
import numpy as np
from ui.utils.AcrylicFlyout import AcrylicFlyoutView, AcrylicFlyout
from ui.utils.TableView import TableViewQWidget
from ui.utils.drawFigure import PlottingThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFileDialog, QGraphicsDropShadowEffect, QFrame, QPushButton
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
import importlib
# from ui.utils.rtspDialog import CustomMessageBox  # 网络摄像头（RTSP）功能已删除
from models import common, yolo, experimental
from ui.utils.webCamera import Camera, WebcamThread  # 本地摄像头功能
from yolocode.yolov8.YOLOv8Thread import YOLOv8Thread
from yolocode.yolov8.YOLOv8SegThread import YOLOv8SegThread
from yolocode.yolov8.YOLOv8PoseThread import YOLOv8PoseThread
from yolocode.yolov11.YOLOv11Thread import YOLOv11Thread
from yolocode.yolov11.YOLOv11SegThread import YOLOv11SegThread
from yolocode.yolov11.YOLOv11PoseThread import YOLOv11PoseThread

glo._init()
glo.set_value('yoloname', "yolov8 yolov11"
                          "yolov8-seg yolov11-seg yolov8-pose yolov8-obb yolov11-obb yolov11-pose"
                          "")

GLOBAL_WINDOW_STATE = True
WIDTH_LEFT_BOX_STANDARD = 180
WIDTH_LEFT_BOX_EXTENDED = 200
WIDTH_SETTING_BAR = 290
WIDTH_LOGO = 60
WINDOW_SPLIT_BODY = 20
KEYS_LEFT_BOX_MENU = ['src_menu', 'src_setting', 'src_webcam', 'src_folder', 'src_vsmode', 'src_setting']  # 已移除 src_camera
# 模型名称和线程类映射
MODEL_THREAD_CLASSES = {
    "yolov8": YOLOv8Thread,
    "yolov11": YOLOv11Thread,
    "yolov8-seg": YOLOv8SegThread,
    "yolov11-seg": YOLOv11SegThread,
    "yolov8-pose": YOLOv8PoseThread,
    "yolov11-pose": YOLOv11PoseThread,
}
# 扩展MODEL_THREAD_CLASSES字典
MODEL_NAME_DICT = list(MODEL_THREAD_CLASSES.items())
for key, value in MODEL_NAME_DICT:
    MODEL_THREAD_CLASSES[f"{key}_left"] = value
    MODEL_THREAD_CLASSES[f"{key}_right"] = value

ALL_MODEL_NAMES = ["yolov8", "yolov11"]
loggertool = LoggerUtils()


# YOLOSHOW窗口类 动态加载UI文件 和 Ui_mainWindow
class YOLOSHOWBASE:
    def __init__(self):
        super().__init__()
        self.inputPath = None
        self.yolo_threads = None
        self.result_statistic = None
        self.detect_result = None
        self.allModelNames = ALL_MODEL_NAMES

    # 初始化左侧菜单栏
    def initSiderWidget(self):
        # --- 侧边栏 --- #
        self.ui.leftBox.setFixedWidth(WIDTH_LEFT_BOX_STANDARD)
        # logo
        self.ui.logo.setFixedSize(WIDTH_LOGO, WIDTH_LOGO)

        # 将左侧菜单栏的按钮固定宽度
        for child_left_box_widget in self.ui.leftbox_bottom.children():

            if isinstance(child_left_box_widget, QFrame):
                child_left_box_widget.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

                for child_left_box_widget_btn in child_left_box_widget.children():
                    if isinstance(child_left_box_widget_btn, QPushButton):
                        child_left_box_widget_btn.setFixedWidth(WIDTH_LEFT_BOX_EXTENDED)

    # 加载模型
    def initModel(self, yoloname=None):
        thread = self.yolo_threads.get(yoloname)
        if not thread:
            raise ValueError(f"No thread found for '{yoloname}'")
        thread.new_model_name = f'{self.current_workpath}/ptfiles/' + self.ui.model_box.currentText()
        thread.progress_value = self.ui.progress_bar.maximum()

        # 信号槽连接使用单独定义的函数，减少闭包的创建
        thread.send_input.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        thread.send_output.connect(lambda x: self.showImg(x, self.ui.main_rightbox, 'img'))
        thread.send_msg.connect(lambda x: self.showStatus(x))
        thread.send_progress.connect(lambda x: self.ui.progress_bar.setValue(x))
        thread.send_fps.connect(lambda x: self.ui.fps_label.setText(str(x)))
        thread.send_class_num.connect(lambda x: self.ui.Class_num.setText(str(x)))
        thread.send_target_num.connect(lambda x: self.ui.Target_num.setText(str(x)))
        thread.send_result_picture.connect(lambda x: self.setResultStatistic(x))
        thread.send_result_table.connect(lambda x: self.setTableResult(x))

    # 阴影效果
    def shadowStyle(self, widget, Color, top_bottom=None):
        shadow = QGraphicsDropShadowEffect(self)
        if 'top' in top_bottom and 'bottom' not in top_bottom:
            shadow.setOffset(0, -5)
        elif 'bottom' in top_bottom and 'top' not in top_bottom:
            shadow.setOffset(0, 5)
        else:
            shadow.setOffset(5, 5)
        shadow.setBlurRadius(10)  # 阴影半径
        shadow.setColor(Color)  # 阴影颜色
        widget.setGraphicsEffect(shadow)

    # 侧边栏缩放
    def scaleMenu(self):
        # standard = 80
        # maxExtend = 180

        leftBoxStart = self.ui.leftBox.width()
        _IS_EXTENDED = leftBoxStart == WIDTH_LEFT_BOX_EXTENDED

        if _IS_EXTENDED:
            leftBoxEnd = WIDTH_LEFT_BOX_STANDARD
        else:
            leftBoxEnd = WIDTH_LEFT_BOX_EXTENDED

        # animation
        self.animation = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.animation.setDuration(500)  # ms
        self.animation.setStartValue(leftBoxStart)
        self.animation.setEndValue(leftBoxEnd)
        self.animation.setEasingCurve(QEasingCurve.InOutQuint)
        self.animation.start()

    # 设置栏缩放
    def scalSetting(self):
        # GET WIDTH
        widthSettingBox = self.ui.settingBox.width()  # right set column width
        widthLeftBox = self.ui.leftBox.width()  # left column length
        maxExtend = WIDTH_SETTING_BAR
        standard = 0

        # SET MAX WIDTH
        if widthSettingBox == 0:
            widthExtended = maxExtend
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                    background-color: qlineargradient(x1:0, y1:0, x2:1 , y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
                                }
                              """)
        else:
            widthExtended = standard
            self.ui.mainbox.setStyleSheet("""
                                  QFrame#mainbox{
                                    border: 1px solid rgba(0, 0, 0, 15%);
                                    border-bottom-left-radius: 0;
                                    border-bottom-right-radius: 0;
                                    border-radius:30%;
                                }
                              """)

        # ANIMATION LEFT BOX
        self.left_box = QPropertyAnimation(self.ui.leftBox, b"minimumWidth")
        self.left_box.setDuration(500)
        self.left_box.setStartValue(widthLeftBox)
        self.left_box.setEndValue(68)
        self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

        # ANIMATION SETTING BOX
        self.setting_box = QPropertyAnimation(self.ui.settingBox, b"minimumWidth")
        self.setting_box.setDuration(500)
        self.setting_box.setStartValue(widthSettingBox)
        self.setting_box.setEndValue(widthExtended)
        self.setting_box.setEasingCurve(QEasingCurve.InOutQuart)

        # SET QSS Change
        self.qss_animation = QPropertyAnimation(self.ui.mainbox, b"styleSheet")
        self.qss_animation.setDuration(300)
        self.qss_animation.setStartValue("""
            QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 white, stop:0.9 #8EC5FC, stop:1 #E0C3FC);
            }
        """)
        self.qss_animation.setEndValue("""
             QFrame#mainbox {
                border: 1px solid rgba(0, 0, 0, 15%);
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-radius:30%;
            }
        """)
        self.qss_animation.setEasingCurve(QEasingCurve.InOutQuart)

        # GROUP ANIMATION
        self.group = QParallelAnimationGroup()
        self.group.addAnimation(self.left_box)
        self.group.addAnimation(self.setting_box)
        self.group.start()

    # 最大化最小化窗口
    def maxorRestore(self):
        global GLOBAL_WINDOW_STATE
        status = GLOBAL_WINDOW_STATE
        if status:
            # 获取当前屏幕的宽度和高度
            self.showMaximized()
            self.ui.maximizeButton.setStyleSheet("""
                          QPushButton:hover{
                               background-color:rgb(139, 29, 31);
                               border-image: url(:/leftbox/images/newsize/scalling.png);
                           }
                      """)
            GLOBAL_WINDOW_STATE = False
        else:
            self.showNormal()
            self.ui.maximizeButton.setStyleSheet("""
                                      QPushButton:hover{
                                           background-color:rgb(139, 29, 31);
                                           border-image: url(:/leftbox/images/newsize/max.png);
                                       }
                                  """)
            GLOBAL_WINDOW_STATE = True

    # 选择照片/视频 并展示
    def selectFile(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/file.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        file_path = config['file_path']
        if not os.path.exists(file_path):
            file_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your Image / Video",  # 标题
            file_path,  # 默认打开路径为当前路径
            "Image / Video type (*.jpg *.jpeg *.png *.bmp *.dib *.jpe *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            self.inputPath = file
            glo.set_value('inputPath', self.inputPath)
            # 如果是视频， 显示第一帧
            if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                # 显示第一帧
                self.cap = cv2.VideoCapture(self.inputPath)
                ret, frame = self.cap.read()
                if ret:
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.showImg(frame, self.ui.main_leftbox, 'img')
            # 如果是图片 正常显示
            else:
                self.showImg(self.inputPath, self.ui.main_leftbox, 'path')
            self.showStatus('已加载文件：{} (Loaded File)'.format(os.path.basename(self.inputPath)))
            config['file_path'] = os.path.dirname(self.inputPath)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 选择摄像头
    def selectWebcam(self):
        try:
            # get the number of local cameras
            cam_num, cams = Camera().get_cam_num()
            if cam_num > 0:
                # popMenu = RoundMenu(parent=self)
                # popMenu.setFixedWidth(self.ui.leftbox_bottom.width())
                # actions = []

                cam = cams[0]
                # cam_name = f'Camera_{cam}'
                # actions.append(Action(cam_name))
                # popMenu.addAction(actions[-1])
                # actions[-1].triggered.connect(lambda: self.actionWebcam(cam))
                self.actionWebcam(cam)

                # x = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).x()
                # y = self.ui.webcamBox.mapToGlobal(self.ui.webcamBox.pos()).y()
                # y = y - self.ui.webcamBox.frameGeometry().height() * 2
                # pos = QPoint(x, y)
                # popMenu.exec(pos, aniType=MenuAnimationType.DROP_DOWN)
            else:
                self.showStatus('未找到摄像头！(No camera found)')
        except Exception as e:
            self.showStatus('%s' % e)

    # 调用网络摄像头
    def actionWebcam(self, cam):
        self.showStatus(f'加载摄像头：Camera_{cam} (Loading camera)')
        self.thread = WebcamThread(cam)
        self.thread.changePixmap.connect(lambda x: self.showImg(x, self.ui.main_leftbox, 'img'))
        self.thread.start()
        self.inputPath = int(cam)

    # 选择文件夹
    def selectFolder(self):
        config_file = f'{self.current_workpath}/config/folder.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        folder_path = config['folder_path']
        if not os.path.exists(folder_path):
            folder_path = os.getcwd()
        FolderPath = QFileDialog.getExistingDirectory(
            self,
            "Select your Folder",
            folder_path  # 起始目录
        )
        if FolderPath:
            FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath) for jpgname in FileFormat
                          if jpgname in filename]
            # self.yolov5_thread.source = Foldername
            self.inputPath = Foldername
            self.showStatus('已加载文件夹：{} (Loaded Folder)'.format(os.path.basename(FolderPath)))
            config['folder_path'] = FolderPath
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

    # 网络摄像头功能已删除
    # 如需使用网络摄像头，请使用本地摄像头功能

    # 显示Label图片
    def showImg(self, img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    # resize 窗口大小
    def resizeGrip(self):
        self.left_grip.setGeometry(0, 10, 10, self.height())
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
        self.top_grip.setGeometry(0, 0, self.width(), 10)
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # 导入模块
    def importModel(self):
        # 获取上次选择文件的路径
        config_file = f'{self.current_workpath}/config/model.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        self.model_path = config['model_path']
        if not os.path.exists(self.model_path):
            self.model_path = os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "Select your YOLO Model",  # 标题
            self.model_path,  # 默认打开路径为当前路径
            "Model File (*.pt)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file:
            fileptPath = os.path.join(self.pt_Path, os.path.basename(file))
            if not os.path.exists(fileptPath):
                shutil.copy(file, self.pt_Path)
                self.showStatus('已加载模型：{} (Loaded Model)'.format(os.path.basename(file)))
                config['model_path'] = os.path.dirname(file)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            else:
                self.showStatus('模型已存在 (Model already exists)')

    # 查看当前模型
    def checkCurrentModel(self, mode=None):
        # 定义模型和对应条件的映射
        model_conditions = {
            "yolov5": lambda name: "yolov5" in name and not self.checkSegName(name),
            "yolov7": lambda name: "yolov7" in name,
            "yolov8": lambda name: "yolov8" in name and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]),
            "yolov9": lambda name: "yolov9" in name,
            "yolov10": lambda name: "yolov10" in name,
            "yolov11": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and not any(
                func(name) for func in [self.checkSegName, self.checkPoseName, self.checkObbName]),
            "rtdetr": lambda name: "rtdetr" in name,
            "yolov5-seg": lambda name: "yolov5" in name and self.checkSegName(name),
            "yolov8-seg": lambda name: "yolov8" in name and self.checkSegName(name),
            "yolov11-seg": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkSegName(name),
            "yolov8-pose": lambda name: "yolov8" in name and self.checkPoseName(name),
            "yolov11-pose": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkPoseName(name),
            "yolov8-obb": lambda name: "yolov8" in name and self.checkObbName(name),
            "yolov11-obb": lambda name: any(sub in name for sub in ["yolov11", "yolo11"]) and self.checkObbName(name),
            "fastsam": lambda name: "fastsam" in name,
            "samv2": lambda name: any(sub in name for sub in ["sam2", "samv2"]),
            "sam": lambda name: "sam" in name
        }

        if mode:
            # VS mode
            model_name = self.model_name1 if mode == "left" else self.model_name2
            model_name = model_name.lower()
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return f"{yoloname}_{mode}"
        else:
            # Single mode
            model_name = self.model_name.lower()
            for yoloname, condition in model_conditions.items():
                if condition(model_name):
                    return yoloname
        return None

    # 检查模型是否符合命名要求
    def checkModelName(self, modelname):
        for name in self.allModelNames:
            if modelname in name:
                return True
        return False

    def checkTaskName(self, modelname, taskname):
        if "yolov5" in modelname:
            return bool(re.match(f'yolo.?5.?-{taskname}.*\.pt$', modelname))
        elif "yolov7" in modelname:
            return bool(re.match(f'yolo.?7.?-{taskname}.*\.pt$', modelname))
        elif "yolov8" in modelname:
            return bool(re.match(f'yolo.?8.?-{taskname}.*\.pt$', modelname))
        elif "yolov9" in modelname:
            return bool(re.match(f'yolo.?9.?-{taskname}.*\.pt$', modelname))
        elif "yolov10" in modelname:
            return bool(re.match(f'yolo.?10.?-{taskname}.*\.pt$', modelname))
        elif "yolo11" in modelname:
            return bool(re.match(f'yolo.?11.?-{taskname}.*\.pt$', modelname))

    # 解决 Modelname 当中的 seg命名问题
    def checkSegName(self, modelname):
        return self.checkTaskName(modelname, "seg")

    # 解决  Modelname 当中的 pose命名问题
    def checkPoseName(self, modelname):
        return self.checkTaskName(modelname, "pose")

    # 解决  Modelname 当中的 pose命名问题
    def checkObbName(self, modelname):
        return self.checkTaskName(modelname, "obb")

    # 停止运行中的模型
    def quitRunningModel(self, stop_status=False):
        for yolo_name in self.yolo_threads.threads_pool.keys():
            try:
                if stop_status:
                    self.yolo_threads.get(yolo_name).stop_dtc = True
                self.yolo_threads.stop_thread(yolo_name)
            except Exception as err:
                loggertool.info(f"Error: {err}")

    # 在MessageBar显示消息
    def showStatus(self, msg):
        self.ui.message_bar.setText(msg)
        if msg == 'Finish Detection' or msg == '检测完成 (Finish Detection)':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            self.ui.progress_bar.setValue(0)
            # self.ui.save_status_button.setEnabled(True)
        elif msg == 'Stop Detection':
            self.quitRunningModel()
            self.ui.run_button.setChecked(False)
            # self.ui.save_status_button.setEnabled(True)
            self.ui.progress_bar.setValue(0)
            self.ui.main_leftbox.clear()  # clear image display
            self.ui.main_rightbox.clear()
            self.ui.Class_num.setText('--')
            self.ui.Target_num.setText('--')
            self.ui.fps_label.setText('--')

    # 导出结果状态判断
    def saveStatus(self):
        self.showStatus('注意：检测结果将被保存 (NOTE: Run image results will be saved)')
        for yolo_thread in self.yolo_threads.threads_pool.values():
            yolo_thread.save_res = True

    # 导出检测结果 --- 过程代码
    def saveResultProcess(self, outdir, current_model_name, folder):
        yolo_thread = self.yolo_threads.get(current_model_name)
        
        # 检查线程是否存在
        if not yolo_thread:
            self.showStatus('未找到模型线程 (Model thread not found)')
            return
        
        # 检查是否有 res_path 属性
        if not hasattr(yolo_thread, 'res_path') or not yolo_thread.res_path:
            self.showStatus('请先运行检测并等待结果生成 (Please run detection first and wait for results)')
            return
        
        if folder:
            try:
                output_dir = os.path.dirname(yolo_thread.res_path)
                if not os.path.exists(output_dir):
                    self.showStatus('请等待结果生成 (Please wait for the result to be generated)')
                    return
                for filename in os.listdir(output_dir):
                    source_path = os.path.join(output_dir, filename)
                    destination_path = os.path.join(outdir, filename)
                    if os.path.isfile(source_path):
                        shutil.copy(source_path, destination_path)
                self.showStatus('保存成功，位置：{} (Saved Successfully)'.format(outdir))
            except Exception as err:
                self.showStatus(f"保存结果时出错：{err} (Error occurred while saving the result)")
        else:
            try:
                if not os.path.exists(yolo_thread.res_path):
                    self.showStatus('请等待结果生成 (Please wait for the result to be generated)')
                    return
                shutil.copy(yolo_thread.res_path, outdir)
                self.showStatus('保存成功，位置：{} (Saved Successfully)'.format(outdir))
            except Exception as err:
                self.showStatus(f"保存结果时出错：{err} (Error occurred while saving the result)")

    def loadAndSetParams(self, config_file, params):
        if not os.path.exists(config_file):
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                params.update(json.load(f))
        return params

    # 加载 Setting 栏
    def loadConfig(self):
        # 1、随机初始化超参数，防止切换模型时，超参数不变
        params = {"iou": round(random.uniform(0, 1), 2),
                  "conf": round(random.uniform(0, 1), 2),
                  "delay": random.randint(10, 50),
                  "line_thickness": random.randint(1, 5)}
        self.updateParams(params)
        # 2、绑定配置项超参数
        params = {"iou": 0.45, "conf": 0.25, "delay": 10, "line_thickness": 3}
        params = self.loadAndSetParams('config/setting.json', params)
        self.updateParams(params)

    # 更新Config超参数
    def updateParams(self, params):
        self.ui.iou_spinbox.setValue(params['iou'])
        self.ui.iou_slider.setValue(int(params['iou'] * 100))
        self.ui.conf_spinbox.setValue(params['conf'])
        self.ui.conf_slider.setValue(int(params['conf'] * 100))
        self.ui.speed_spinbox.setValue(params['delay'])
        self.ui.speed_slider.setValue(params['delay'])
        self.ui.line_spinbox.setValue(params['line_thickness'])
        self.ui.line_slider.setValue(params['line_thickness'])

    # 加载 pt 模型到 model_box
    def loadModels(self):
        pt_list = os.listdir(f'{self.current_workpath}/ptfiles/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(f'{self.current_workpath}/ptfiles/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ui.model_box.clear()
            self.ui.model_box.addItems(self.pt_list)

    # 重载模型
    def reloadModel(self):
        importlib.reload(common)
        importlib.reload(yolo)
        importlib.reload(experimental)

    def use_mp(self, use_mp):
        for yolo_thread in self.yolo_threads.threads_pool.values():
            yolo_thread.use_mp = use_mp

    # 调整超参数
    def changeValue(self, x, flag):
        if flag == 'iou_spinbox':
            self.ui.iou_slider.setValue(int(x * 100))  # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.ui.iou_spinbox.setValue(x / 100)  # The slider value changes, changing the box
            self.showStatus('IOU 交并比阈值: %s (IOU Threshold)' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.ui.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.ui.conf_spinbox.setValue(x / 100)
            self.showStatus('置信度阈值: %s (Conf Threshold)' % str(x / 100))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.ui.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.ui.speed_spinbox.setValue(x)
            self.showStatus('延迟: %s ms (Delay)' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.speed_thres = x  # ms
        elif flag == 'line_spinbox':
            self.ui.line_slider.setValue(x)
        elif flag == 'line_slider':
            self.ui.line_spinbox.setValue(x)
            self.showStatus('线宽: %s (Line Width)' % str(x))
            for yolo_thread in self.yolo_threads.threads_pool.values():
                yolo_thread.line_thickness = x

    # 修改YOLOv5、YOLOv7、YOLOv9 解决 yolo.py冲突
    def solveYoloConflict(self, ptnamelst):
        for ptname in ptnamelst:
            ptbaseName = os.path.basename(ptname)
            if "yolov5" in ptbaseName and not self.checkSegName(ptbaseName):
                glo.set_value('yoloname', "yolov5")
                self.reloadModel()
                from models.yolo import Detect_YOLOV5
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov5_detect = Detect_YOLOV5(anchors=_anchors, nc=_nc, ch=_ch)
                        yolov5_detect.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov5_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov5_detect
                torch.save(net, ptname)
            elif "yolov5" in ptbaseName and self.checkSegName(ptbaseName):
                glo.set_value('yoloname', "yolov5-seg")
                self.reloadModel()
                from models.yolo import Segment_YOLOV5
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Segment':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_seg_layers = _yaml_lst[-1][0]
                        for layer in _yaml_seg_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov5_seg = Segment_YOLOV5(anchors=_anchors, nc=_nc, ch=_ch)
                        _module.detect = yolov5_seg.detect
                        yolov5_seg.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov5_seg.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov5_seg
                torch.save(net, ptname)
            elif "yolov7" in ptbaseName:
                glo.set_value('yoloname', "yolov7")
                self.reloadModel()
                from models.yolo import Detect_YOLOV7
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _anchors = _module.anchors
                        _nc = _module.nc
                        yolov7_detect = Detect_YOLOV7(anchors=_anchors, nc=_nc, ch=_ch)
                        yolov7_detect.__dict__.update(_module.__dict__)
                        for _new_param, _old_param in zip(yolov7_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov7_detect
                torch.save(net, ptname)
            elif "yolov9" in ptbaseName:
                glo.set_value('yoloname', "yolov9")
                self.reloadModel()
                from models.yolo import Detect_YOLOV9
                net = torch.load(ptname)
                for _module_index in range(len(net['model'].model)):
                    _module = net['model'].model[_module_index]
                    _module_name = _module.__class__.__name__
                    if _module_name == 'Detect':
                        _yaml_lst = net['model'].yaml['backbone'] + net['model'].yaml['head']
                        _ch = []
                        _yaml_detect_layers = _yaml_lst[-1][0]
                        for layer in _yaml_detect_layers:
                            _ch.append(_yaml_lst[layer][-1][0])
                        _nc = _module.nc
                        yolov9_detect = Detect_YOLOV9(nc=_nc, ch=_ch)
                        for _new_param, _old_param in zip(yolov9_detect.parameters(), _module.parameters()):
                            _new_param.data = _old_param.data.clone()
                        net['model'].model[_module_index] = yolov9_detect
                torch.save(net, ptname)
        glo.set_value("yoloname", "yolov5 yolov8 yolov5-seg yolov8-seg yolov8-pose")
        self.reloadModel()

    # 接受统计结果，然后写入json中
    def setResultStatistic(self, value):
        # 写入 JSON 文件
        with open('config/result.json', 'w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=4)
        # --- 获取统计结果 + 绘制柱状图 --- #
        self.result_statistic = value
        self.plot_thread = PlottingThread(self.result_statistic, self.current_workpath)
        self.plot_thread.start()
        # --- 获取统计结果 + 绘制柱状图 --- #

    # 展示柱状图结果
    def showResultStatics(self):
        self.resutl_statistic = dict()
        # 读取 JSON 文件
        result_json_path = self.current_workpath + r'\config\result.json'
        result_png_path = self.current_workpath + r'\config\result.png'
        
        # 检查文件是否存在
        if not os.path.exists(result_json_path):
            view = AcrylicFlyoutView(
                title='结果统计 (Result Statistics)',
                content="未检测到已完成的目标检测结果，请先执行检测任务！\n(No completed target detection results detected, please execute the detection task first!)",
                isClosable=True
            )
        else:
            with open(result_json_path, 'r', encoding='utf-8') as file:
                self.result_statistic = json.load(file)
            
            # 检查结果是否为空或result.png是否存在
            if self.result_statistic and os.path.exists(result_png_path):
                # 创建新字典，使用中文键
                result_str = ""
                for index, (key, value) in enumerate(self.result_statistic.items()):
                    result_str += f"{key}:{value}x \t"
                    if (index + 1) % 4 == 0:
                        result_str += "\n"

                view = AcrylicFlyoutView(
                    title='检测目标类别分布（百分比）(Detected Target Category Distribution)',
                    content=result_str,
                    image=result_png_path,
                    isClosable=True
                )
            else:
                # 区分两种情况：未检测 vs 检测完成但没有目标
                view = AcrylicFlyoutView(
                    title='结果统计 (Result Statistics)',
                    content="检测已完成，但未检测到任何目标（Classes=0, Targets=0）\n"
                           "可能原因：\n"
                           "• 图片/视频中没有人物\n"
                           "• 置信度阈值设置过高，尝试降低 Confidence 参数\n"
                           "• 模型不适合当前场景，尝试更换模型\n\n"
                           "(Detection completed, but no targets detected)\n"
                           "Possible reasons:\n"
                           "• No person in the image/video\n"
                           "• Confidence threshold too high, try lowering it\n"
                           "• Model not suitable for current scene",
                    isClosable=True
                )

        # 修改字体大小
        view.titleLabel.setStyleSheet("""font-size: 30px; 
                                            color: black; 
                                            font-weight: bold; 
                                            font-family: 'KaiTi';
                                        """)
        view.contentLabel.setStyleSheet("""font-size: 25px; 
                                            color: black; 
                                            font-family: 'KaiTi';""")
        # 修改image的大小
        width = self.ui.rightbox_main.width() // 2.5
        height = self.ui.rightbox_main.height() // 2.5
        view.imageLabel.setFixedSize(width, height)
        # adjust layout (optional)
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)

        # show view
        w = AcrylicFlyout.make(view, self.ui.rightbox_play, self)
        view.closed.connect(w.close)

    # 获取表格结果的列表
    def setTableResult(self, value):
        self.detect_result = value

    # 展示表格结果
    def showTableResult(self):
        # 检查是否有检测结果
        if self.detect_result is None or len(self.detect_result) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "无检测结果 No Results",
                "暂无检测结果！\n\n"
                "📹 摄像头模式说明：\n"
                "• 摄像头检测是实时流模式\n"
                "• 需要在检测过程中积累数据\n"
                "• 停止检测后才能查看统计结果\n\n"
                "✅ 正确操作流程：\n"
                "1️⃣ 点击 'Webcam' 启动摄像头\n"
                "2️⃣ 点击 'Run' 开始检测\n"
                "3️⃣ 保持检测运行至少20-30秒\n"
                "4️⃣ 确保摄像头画面中有人\n"
                "5️⃣ 点击 'Run' 停止检测\n"
                "6️⃣ 点击 'Result Tab' 查看结果\n\n"
                "💡 如果仍然没有数据：\n"
                "• 确保摄像头画面中有人\n"
                "• 降低置信度阈值（Conf Threshold）\n"
                "• 检测时间要足够长（至少20秒）\n"
                "• 尝试不同的坐姿让系统检测\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📹 Webcam Mode:\n"
                "• Real-time streaming mode\n"
                "• Data accumulates during detection\n"
                "• View results after stopping\n\n"
                "✅ Correct Steps:\n"
                "1️⃣ Click 'Webcam' to start camera\n"
                "2️⃣ Click 'Run' to start detection\n"
                "3️⃣ Keep detection running for 20-30s\n"
                "4️⃣ Ensure person is in camera view\n"
                "5️⃣ Click 'Run' to stop detection\n"
                "6️⃣ Click 'Result Tab' to view results\n\n"
                "💡 If still no data:\n"
                "• Ensure person is visible in camera\n"
                "• Lower confidence threshold\n"
                "• Run detection longer (at least 20s)\n"
                "• Try different postures"
            )
            return
        
        # 关闭旧窗口（如果存在）
        if hasattr(self, 'table_result') and self.table_result:
            try:
                self.table_result.close()
            except:
                pass
        
        # 创建并显示新窗口
        self.table_result = TableViewQWidget(infoList=self.detect_result)
        self.table_result.show()
