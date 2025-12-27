from utils import glo
import json
import os
import cv2
from PySide6.QtGui import QMouseEvent, QGuiApplication
from PySide6.QtCore import Qt, QPropertyAnimation, Signal
from ui.utils.customGrips import CustomGrip
from yoloshow.YOLOSHOW import YOLOSHOW
from yoloshow.YOLOSHOWVS import YOLOSHOWVS


class YOLOSHOWWindow(YOLOSHOW):
    # 定义关闭信号
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWWindow, self).__init__()
        self.setup_tooltips()
        self.center()
        # --- 拖动窗口 改变窗口大小 --- #
    
    def setup_tooltips(self):
        """设置所有控件的工具提示"""
        # 为spinbox和slider添加悬停高亮样式
        hover_style_spinbox = """
        QDoubleSpinBox {
            border: 0px solid lightgray;
            border-radius: 2px;
            background-color: rgba(255,255,255,90);
            font: 600 9pt "Segoe UI";
        }
        QDoubleSpinBox:hover {
            background-color: rgba(114, 129, 214, 59);
            border: 1px solid rgba(114, 129, 214, 150);
        }
        QDoubleSpinBox::up-button {
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(:/setting /images/newsize/box_up.png);
        }
        QDoubleSpinBox::up-button:pressed {
            margin-top: 1px;
        }
        QDoubleSpinBox::down-button {
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image:url(:/setting /images/newsize/box_down.png);
        }
        QDoubleSpinBox::down-button:pressed {
            margin-bottom: 1px;
        }
        """
        
        hover_style_spinbox_int = """
        QSpinBox {
            border: 0px solid lightgray;
            border-radius: 2px;
            background-color: rgba(255,255,255,90);
            font: 600 9pt "Segoe UI";
        }
        QSpinBox:hover {
            background-color: rgba(114, 129, 214, 59);
            border: 1px solid rgba(114, 129, 214, 150);
        }
        QSpinBox::up-button {
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image: url(:/setting /images/newsize/box_up.png);
        }
        QSpinBox::up-button:pressed {
            margin-top: 1px;
        }
        QSpinBox::down-button {
            width: 10px;
            height: 9px;
            margin: 0px 3px 0px 0px;
            border-image:url(:/setting /images/newsize/box_down.png);
        }
        QSpinBox::down-button:pressed {
            margin-bottom: 1px;
        }
        """
        
        hover_style_slider = """
        QSlider::groove:horizontal {
            border: none;
            height: 10px;
            background-color: rgba(255,255,255,90);
            border-radius: 5px;
        }
        QSlider::groove:horizontal:hover {
            background-color: rgba(114, 129, 214, 30);
        }
        QSlider::handle:horizontal {
            width: 10px;
            margin: -1px 0px -1px 0px;
            border-radius: 3px;
            background-color: white;
        }
        QSlider::handle:horizontal:hover {
            background-color: rgba(114, 129, 214, 200);
        }
        QSlider::sub-page:horizontal {
            background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1, stop:0 rgb(253, 139, 133), stop:1 rgb(248, 194, 152));
            border-radius: 5px;
        }
        """
        
        # 应用悬停样式
        self.ui.iou_spinbox.setStyleSheet(hover_style_spinbox)
        self.ui.conf_spinbox.setStyleSheet(hover_style_spinbox)
        self.ui.line_spinbox.setStyleSheet(hover_style_spinbox)
        self.ui.speed_spinbox.setStyleSheet(hover_style_spinbox_int)
        
        self.ui.iou_slider.setStyleSheet(hover_style_slider)
        self.ui.conf_slider.setStyleSheet(hover_style_slider)
        self.ui.speed_slider.setStyleSheet(hover_style_slider)
        self.ui.line_slider.setStyleSheet(hover_style_slider)
        
        # Model 模型选择
        self.ui.model_box.setToolTip(
            "【模型选择】\n"
            "• yolov11-eq.pt - 坐姿检测专用模型（推荐）⭐\n"
            "  训练了6种坐姿：正确坐姿、身体左倾、身体右倾、左手托腮、右手托腮、趴桌\n"
            "• yolo11n.pt - 通用目标检测模型\n"
            "• yolo11n-pose.pt - 人体姿态检测模型\n"
            "• yolo11n-seg.pt - 图像分割模型\n\n"
            "💡 建议：使用 yolov11-eq.pt 获得最佳坐姿检测效果")
        
        # Mediapipe 骨骼提取
        self.ui.mp_button.setToolTip(
            "【Mediapipe 骨骼提取】\n"
            "作用：提取人体骨骼关键点，辅助坐姿检测\n"
            "• 开启：提高检测精度，更准确识别坐姿（推荐）⭐\n"
            "• 关闭：仅使用YOLO检测，速度更快但精度略低\n\n"
            "首次使用：需要下载模型文件（约10分钟，仅一次）\n"
            "后续使用：直接加载本地缓存，启动很快\n\n"
            "💡 建议：保持开启状态以获得最佳检测效果")
        
        # IOU 交并比
        self.ui.iou_spinbox.setToolTip(
            "【IOU 交并比阈值】\n"
            "作用：控制重叠检测框的合并程度\n"
            "• 数值范围：0.01 - 1.00\n"
            "• 默认值：0.45（推荐 0.45-0.50）\n\n"
            "调整效果：\n"
            "• 调高（0.50-0.70）：保留更多重叠框，可能出现重复检测\n"
            "• 调低（0.30-0.45）：合并更多重叠框，减少重复检测\n\n"
            "💡 建议：保持默认值 0.45，一般无需调整")
        
        self.ui.iou_slider.setToolTip(
            "【IOU 交并比阈值】\n"
            "拖动滑块快速调整 IOU 值\n"
            "推荐范围：0.45-0.50")
        
        # Confidence 置信度
        self.ui.conf_spinbox.setToolTip(
            "【Confidence 置信度阈值】\n"
            "作用：控制检测的灵敏度，过滤低置信度的检测结果\n"
            "• 数值范围：0.01 - 1.00\n"
            "• 默认值：0.25（推荐 0.25-0.35）\n\n"
            "调整效果：\n"
            "• 调高（0.35-0.50）：只显示高置信度结果，减少误检但可能漏检\n"
            "  适用场景：误检太多、需要更准确的结果\n"
            "• 调低（0.15-0.25）：显示更多检测结果，提高检出率但可能误检\n"
            "  适用场景：检测不到目标、需要更灵敏的检测\n\n"
            "💡 建议：\n"
            "  - 检测不到 → 降低到 0.20-0.25\n"
            "  - 误检太多 → 提高到 0.35-0.40\n"
            "  - 正常使用 → 保持 0.25-0.30")
        
        self.ui.conf_slider.setToolTip(
            "【Confidence 置信度阈值】\n"
            "拖动滑块快速调整置信度\n"
            "推荐范围：0.25-0.35")
        
        # Delay 延迟
        self.ui.speed_spinbox.setToolTip(
            "【Delay 帧间延迟】\n"
            "作用：控制视频/摄像头处理的帧间延迟时间\n"
            "• 数值范围：0 - 50 毫秒\n"
            "• 默认值：10ms（推荐 5-10ms）\n\n"
            "调整效果：\n"
            "• 调高（15-30ms）：降低处理速度，减少CPU/GPU占用\n"
            "  适用场景：电脑卡顿、风扇噪音大、需要节省资源\n"
            "• 调低（1-5ms）：提高处理速度，更流畅但占用更多资源\n"
            "  适用场景：性能充足、需要实时性更高的检测\n"
            "• 设为0：最快速度，但可能导致系统卡顿\n\n"
            "💡 建议：\n"
            "  - 电脑卡顿 → 增加到 15-20ms\n"
            "  - 性能充足 → 保持 5-10ms\n"
            "  - 处理视频文件 → 可设为 1-5ms")
        
        self.ui.speed_slider.setToolTip(
            "【Delay 帧间延迟】\n"
            "拖动滑块快速调整延迟时间\n"
            "推荐范围：5-10ms")
        
        # Line Width 线宽
        self.ui.line_spinbox.setToolTip(
            "【Line Width 检测框线宽】\n"
            "作用：调整检测框边框的粗细程度（仅影响显示效果）\n"
            "• 数值范围：0 - 5 像素\n"
            "• 默认值：3（推荐 2-3）\n\n"
            "调整效果：\n"
            "• 调高（3-5）：边框更粗，更容易看清检测框\n"
            "• 调低（1-2）：边框更细，画面更简洁\n"
            "• 设为0：不显示边框（不推荐）\n\n"
            "💡 说明：此参数不影响检测性能，仅改变视觉效果")
        
        self.ui.line_slider.setToolTip(
            "【Line Width 检测框线宽】\n"
            "拖动滑块快速调整线宽\n"
            "推荐范围：2-3")
        
        # Save Result 保存结果
        self.ui.save_button.setToolTip(
            "【Save Result 保存检测结果】\n"
            "作用：将检测结果保存到本地文件\n\n"
            "⚠️ 使用条件：\n"
            "• 仅适用于图片/视频文件检测\n"
            "• 必须等待检测完成（显示'检测完成'）\n"
            "• 不支持摄像头/网络摄像头实时检测\n\n"
            "保存内容：\n"
            "• 标注后的图片/视频（带检测框和标签）\n\n"
            "💡 提示：\n"
            "  - 摄像头检测请使用截图功能保存\n"
            "  - 结果图和结果表会在检测完成时自动生成")
        
        # 左侧菜单项工具提示
        self.ui.src_img.setToolTip(
            "【Media 媒体】\n"
            "选择图片或视频文件进行检测\n\n"
            "支持格式：\n"
            "• 图片：jpg, png, bmp, jpeg 等\n"
            "• 视频：mp4, avi, mkv 等\n\n"
            "特点：\n"
            "• 检测完成后可使用 Save Result\n"
            "• 自动生成结果图和结果表\n"
            "• 支持保存带标注的文件\n\n"
            "💡 提示：适合对单个文件进行详细分析")
        
        self.ui.src_webcam.setToolTip(
            "【Webcam 摄像头】⭐\n"
            "使用本地摄像头进行实时坐姿检测\n\n"
            "特点：\n"
            "• 实时检测，持续运行\n"
            "• 可查看结果图和结果表\n"
            "• 不支持 Save Result 功能\n\n"
            "💡 建议：\n"
            "  - 推荐用于日常坐姿监测\n"
            "  - 需要保存请使用截图功能")
        
        self.ui.src_folder.setToolTip(
            "【Folder 文件夹】\n"
            "批量处理文件夹中的所有图片/视频\n\n"
            "特点：\n"
            "• 自动处理文件夹内所有文件\n"
            "• 检测完成后可使用 Save Result\n"
            "• 自动生成结果图和结果表\n"
            "• 支持批量保存检测结果\n\n"
            "💡 提示：适合大量文件的批处理分析")
        
        self.ui.src_camera.setToolTip(
            "【IPcam 网络摄像头】\n"
            "连接网络摄像头或RTSP视频流\n\n"
            "支持协议：\n"
            "• RTSP 流（rtsp://...）\n"
            "• HTTP 流（http://...）\n\n"
            "特点：\n"
            "• 实时检测，持续运行\n"
            "• 可查看结果图和结果表\n"
            "• 不支持 Save Result 功能\n\n"
            "💡 提示：需要输入完整的流地址")
        
        self.ui.src_result.setToolTip(
            "【Result Pic 结果图片】\n"
            "查看检测结果的统计柱状图\n\n"
            "📊 统计内容：\n"
            "• X轴：检测到的类别（如正确坐姿、前倾等）\n"
            "• Y轴：每个类别的占比百分比\n"
            "• 柱子上方：精确的百分比数值\n\n"
            "📈 数据来源：\n"
            "• 图片/视频：统计整个文件的检测结果\n"
            "• 摄像头：累计统计所有帧的检测结果\n\n"
            "⚠️ 注意：\n"
            "• 检测完成后自动生成\n"
            "• 不需要点击 Save Result\n"
            "• 支持所有检测模式\n\n"
            "💡 示例：正确坐姿 45%，前倾 30%，后仰 25%")
        
        self.ui.src_table.setToolTip(
            "【Result Tab 结果表格】\n"
            "查看检测统计数据表格\n\n"
            "📋 显示内容：\n"
            "• 类别名称：如正确坐姿、前倾、后仰等\n"
            "• 检测数量：每个类别检测到的次数\n\n"
            "📈 数据来源：\n"
            "• 图片/视频：统计整个文件的检测结果\n"
            "• 摄像头：累计统计所有帧的检测结果\n\n"
            "⚠️ 注意：\n"
            "• 检测完成后自动生成\n"
            "• 不需要点击 Save Result\n"
            "• 支持所有检测模式\n\n"
            "💡 示例：正确坐姿 45次，前倾 30次，后仰 25次")
        
        self.ui.src_vsmode.setToolTip(
            "【VS Mode 对比模式】\n"
            "同时使用两个模型进行对比检测\n"
            "可以比较不同模型的检测效果")
        
        # 状态栏工具提示 - 简化版本，只保留名词解释
        # 为标签添加悬停高亮样式
        hover_style_label = """
        QLabel {
            color: white;
            font: 600 9pt "Segoe UI";
        }
        QLabel:hover {
            background-color: rgba(114, 129, 214, 100);
            border-radius: 3px;
            padding: 2px;
        }
        """
        
        self.ui.label_5.setStyleSheet(hover_style_label)
        self.ui.label_6.setStyleSheet(hover_style_label)
        self.ui.label_7.setStyleSheet(hover_style_label)
        self.ui.label_8.setStyleSheet(hover_style_label)
        
        self.ui.label_5.setToolTip("Classes 类别：显示检测到的不同类别数量")
        self.ui.label_6.setToolTip("Targets 目标：显示检测到的目标总数")
        self.ui.label_7.setToolTip("Fps 帧率：显示每秒处理的帧数")
        self.ui.label_8.setToolTip("Model 模型：显示当前使用的检测模型")
        
        # 为数值标签也添加悬停高亮样式
        hover_style_value = """
        QLabel {
            color: white;
            font: 600 9pt "Segoe UI";
        }
        QLabel:hover {
            background-color: rgba(114, 129, 214, 100);
            border-radius: 3px;
            padding: 2px;
        }
        """
        
        self.ui.Class_num.setStyleSheet(hover_style_value)
        self.ui.Target_num.setStyleSheet(hover_style_value)
        self.ui.fps_label.setStyleSheet(hover_style_value)
        self.ui.Model_label.setStyleSheet(hover_style_value)
        
        self.ui.Class_num.setToolTip("Classes 类别：显示检测到的不同类别数量")
        self.ui.Target_num.setToolTip("Targets 目标：显示检测到的目标总数")
        self.ui.fps_label.setToolTip("Fps 帧率：显示每秒处理的帧数")
        self.ui.Model_label.setToolTip("Model 模型：显示当前使用的检测模型")
    
    def dragEnterEvent(self, event):
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True)  # ==> 设置窗口支持拖动（必须设置）
        # --- 拖动窗口 改变窗口大小 --- #
        self.animation_window = None

    # 鼠标拖入事件
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 检查是否为文件
            event.acceptProposedAction()  # 接受拖拽的数据


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 获取所有文件路径
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 获取文件路径
        if file:
            # 判断是否是文件夹
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 显示文件夹中第一张图片
                self.showStatus('已加载文件夹：{} (Loaded Folder)'.format(os.path.basename(file)))
            # 图片 / 视频
            else:
                self.inputPath = file
                # 如果是视频， 显示第一帧
                if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                    # 显示第一帧
                    self.cap = cv2.VideoCapture(self.inputPath)
                    ret, frame = self.cap.read()
                    if ret:
                        # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.showImg(frame, self.main_leftbox, 'img')
                # 如果是图片 正常显示
                else:
                    self.showImg(self.inputPath, self.main_leftbox, 'path')
                self.showStatus('已加载文件：{} (Loaded File)'.format(os.path.basename(self.inputPath)))
        glo.set_value('inputPath', self.inputPath)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6获取屏幕参数
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)

    # 拖动窗口 改变窗口大小
    def resizeEvent(self, event):
        # Update Size Grips
        self.resizeGrip()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 这里定义显示动画
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 动画时间500毫秒
            self.animation.setStartValue(0)  # 从完全透明开始
            self.animation.setEndValue(1)  # 到完全不透明结束
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()

# 多套一个类 为了实现MouseLabel方法
class YOLOSHOWVSWindow(YOLOSHOWVS):
    closed = Signal()

    def __init__(self):
        super(YOLOSHOWVSWindow, self).__init__()
        self.center()
        # --- 拖动窗口 改变窗口大小 --- #
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
        self.right_grip = CustomGrip(self, Qt.RightEdge, True)
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)
        self.setAcceptDrops(True) # ==> 设置窗口支持拖动（必须设置）
        # --- 拖动窗口 改变窗口大小 --- #
        self.animation_window = None


    # 鼠标拖入事件
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # 检查是否为文件
            event.acceptProposedAction()  # 接受拖拽的数据


    def dropEvent(self, event):
        # files = [url.toLocalFile() for url in event.mimeData().urls()]  # 获取所有文件路径
        file = event.mimeData().urls()[0].toLocalFile()  # ==> 获取文件路径
        if file:
            # 判断是否是文件夹
            if os.path.isdir(file):
                FileFormat = [".mp4", ".mkv", ".avi", ".flv", ".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2"]
                Foldername = [(file + "/" + filename) for filename in os.listdir(file) for jpgname in
                              FileFormat
                              if jpgname in filename]
                self.inputPath = Foldername
                self.showImg(self.inputPath[0], self.main_leftbox, 'path')  # 显示文件夹中第一张图片
                self.showStatus('已加载文件夹：{} (Loaded Folder)'.format(os.path.basename(file)))
            # 图片 / 视频
            else:
                self.inputPath = file
                # 如果是视频， 显示第一帧
                if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                    # 显示第一帧
                    self.cap = cv2.VideoCapture(self.inputPath)
                    ret, frame = self.cap.read()
                    if ret:
                        # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.showImg(frame, self.main_leftbox, 'img')
                # 如果是图片 正常显示
                else:
                    self.showImg(self.inputPath, self.main_leftbox, 'path')
                self.showStatus('已加载文件：{} (Loaded File)'.format(os.path.basename(self.inputPath)))
        glo.set_value('inputPath', self.inputPath)


    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6获取屏幕参数
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)

    # 拖动窗口 改变窗口大小
    def resizeEvent(self, event):
        # Update Size Grips
        self.resizeGrip()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            # 这里定义显示动画
            self.animation = QPropertyAnimation(self, b"windowOpacity")
            self.animation.setDuration(500)  # 动画时间500毫秒
            self.animation.setStartValue(0)  # 从完全透明开始
            self.animation.setEndValue(1)  # 到完全不透明结束
            self.animation.start()

    def closeEvent(self, event):
        if not self.animation_window:
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.ui.iou_spinbox.value()
            config['conf'] = self.ui.conf_spinbox.value()
            config['delay'] = self.ui.speed_spinbox.value()
            config['line_thickness'] = self.ui.line_spinbox.value()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.animation_window = QPropertyAnimation(self, b"windowOpacity")
            self.animation_window.setStartValue(1)
            self.animation_window.setEndValue(0)
            self.animation_window.setDuration(500)
            self.animation_window.start()
            self.animation_window.finished.connect(self.close)
            event.ignore()
        else:
            self.setWindowOpacity(1.0)
            self.closed.emit()