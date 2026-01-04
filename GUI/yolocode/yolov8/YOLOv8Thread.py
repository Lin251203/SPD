import os.path
import time

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path
from yolocode.yolov8.data import load_inference_source
from yolocode.yolov8.data.augment import classify_transforms, LetterBox
from yolocode.yolov8.data.utils import IMG_FORMATS, VID_FORMATS
from yolocode.yolov8.engine.predictor import STREAM_WARNING
from yolocode.yolov8.engine.results import Results
from models.common import AutoBackend
from yolocode.yolov8.utils import callbacks, ops, LOGGER, colorstr, MACOS, WINDOWS
from collections import defaultdict
from yolocode.yolov8.utils.checks import check_imgsz, increment_path
from yolocode.yolov8.utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor
# 加入mediapipe v1.1
import mediapipe as mp


class YOLOv8Thread(QThread):
    # 输入 输出 消息
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_msg = Signal(str)
    # 状态栏显示数据 进度条数据
    send_fps = Signal(str)  # fps
    # send_labels = Signal(dict)  # Detected target results (number of each category)
    send_progress = Signal(int)  # Completeness
    send_class_num = Signal(int)  # Number of categories detected
    send_target_num = Signal(int)  # Targets detected
    send_result_picture = Signal(dict)  # Send the result picture
    send_result_table = Signal(list)  # Send the result table

    def __init__(self):
        super(YOLOv8Thread, self).__init__()
        # YOLOSHOW 界面参数设置
        self.ori_img = None
        self.results = None
        self.current_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = None  # input source
        self.stop_dtc = True  # 停止检测
        self.is_continue = True  # continue/pause
        self.save_res = False  # Save test results
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.all_labels_dict = {}  # return a dictionary of all results(only for video)
        self.progress_value = 0  # progress bar
        self.res_status = False  # result status
        self.parent_workpath = None  # parent work path
        self.executor = ThreadPoolExecutor(max_workers=1)  # 只允许一个线程运行

        # mediapipe 参数设置
        self.use_mp = False  # 是否使用mediapipe显示骨骼和手部
        self.mp_pose = None  # mediapipe pose
        self.mp_pose_results = None  # mediapipe pose results

        # YOLOv8 参数设置
        self.model = None
        self.data = 'yolocode/yolov8/cfg/datasets/coco.yaml'  # data_dict
        self.imgsz = 640
        self.device = ''
        self.dataset = None
        self.task = 'detect'
        self.dnn = False
        self.half = False
        self.agnostic_nms = False
        self.stream_buffer = False
        self.crop_fraction = 1.0
        self.done_warmup = False
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None
        self.batch = None
        self.batchsize = 1
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 视频帧率
        self.max_det = 1000  # 最大检测数
        self.classes = None  # 指定检测类别
        self.line_thickness = 3
        self.results_picture = dict()  # 结果图片
        self.results_table = list()  # 结果表格
        self.file_path = None  # 文件路径
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    def run(self):
        # 创建日志文件夹
        import datetime
        import os
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # 创建日志文件
        log_file_path = os.path.join(debug_dir, f"debug_thread_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        try:
            log_file = open(log_file_path, 'w', encoding='utf-8')
            
            def thread_log(message):
                """写入日志到文件和控制台"""
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                log_message = f"[{timestamp}] {message}"
                print(log_message)
                try:
                    log_file.write(log_message + '\n')
                    log_file.flush()
                except:
                    pass
            
            thread_log(f"[DEBUG] ========== YOLOv8Thread.run() 开始 ==========")
            thread_log(f"[DEBUG] model={self.model}, new_model_name={self.new_model_name}")

            if not self.model:
                thread_log(f"[DEBUG] 模型未加载，开始加载模型: {self.new_model_name}")
                self.send_msg.emit("正在加载模型：{} (Loading model)".format(os.path.basename(self.new_model_name)))
                thread_log(f"[DEBUG] 调用 setup_model")
                try:
                    self.setup_model(self.new_model_name)
                    thread_log(f"[DEBUG] setup_model 完成")
                except Exception as e:
                    thread_log(f"[ERROR] setup_model 出错: {e}")
                    import traceback
                    error_trace = traceback.format_exc()
                    thread_log(error_trace)
                    # 发送错误消息到UI
                    self.send_msg.emit(f"模型加载失败: {str(e)} (Model loading failed)")
                    # 不要 raise，避免崩溃，直接返回
                    return
                self.used_model_name = self.new_model_name
                thread_log(f"[DEBUG] 模型加载完成")

            thread_log(f"[DEBUG] 开始处理输入源: {self.source}")
            source = str(self.source)
            # 判断输入源类型
            if isinstance(IMG_FORMATS, str) or isinstance(IMG_FORMATS, tuple):
                self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            else:
                self.is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
            self.is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
            self.webcam = source.isnumeric() or source.endswith(".streams") or (self.is_url and not self.is_file)
            self.screenshot = source.lower().startswith("screen")
            # 判断输入源是否是文件夹，如果是列表，则是文件夹
            self.is_folder = isinstance(self.source, list)
            if self.save_res:
                self.save_path = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
                self.save_path.mkdir(parents=True, exist_ok=True)  # make dir

            if self.is_folder:
                # 文件夹检测：在开始检测整个文件夹前清空累积结果
                self.all_labels_dict = {}
                print(f"[DEBUG] 文件夹检测：已清空累积结果 all_labels_dict")
                
                for index, source in enumerate(self.source):
                    is_folder_last = True if index + 1 == len(self.source) else False
                    self.setup_source(source)
                    self.detect(is_folder_last=is_folder_last)
            else:
                self.setup_source(source)
                self.go_process()
                self.detect()
                
            thread_log(f"[DEBUG] ========== YOLOv8Thread.run() 完成 ==========")
        except Exception as e:
            if 'log_file' in locals():
                thread_log(f"[ERROR] YOLOv8Thread.run() 出错: {e}")
                import traceback
                thread_log(traceback.format_exc())
            raise
        finally:
            if 'log_file' in locals():
                try:
                    log_file.close()
                except:
                    pass

    def go_process(self):
        for i in range(0, 101, 10):
            self.send_progress.emit(i)

    @torch.no_grad()
    def detect(self, is_folder_last=False):
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        datasets = iter(self.dataset)
        count = 0
        start_time = time.time()  # used to calculate the frame rate
        
        # 清空累积的检测结果，避免不同模型或不同检测任务之间的结果累加
        # 注意：文件夹检测时，不在这里清空，而是在 run() 方法中清空一次
        if not self.is_folder:
            self.all_labels_dict = {}
            print(f"[DEBUG] 已清空累积结果 all_labels_dict")
        while True:
            if self.stop_dtc:
                if self.is_folder and not is_folder_last:
                    break
                
                # === 调试信息 ===
                print(f"[DEBUG] 停止检测")
                print(f"[DEBUG] self.webcam = {self.webcam}")
                print(f"[DEBUG] hasattr all_labels_dict = {hasattr(self, 'all_labels_dict')}")
                print(f"[DEBUG] all_labels_dict = {getattr(self, 'all_labels_dict', 'NOT_FOUND')}")
                print(f"[DEBUG] results_picture (before) = {self.results_picture}")
                
                self.send_msg.emit('停止检测 (Stop Detection)')
                
                # --- 摄像头模式：使用累积的结果 --- #
                if self.webcam and hasattr(self, 'all_labels_dict') and self.all_labels_dict:
                    self.results_picture = self.all_labels_dict.copy()
                    print(f"[摄像头模式] 累积检测结果: {self.results_picture}")
                else:
                    print(f"[WARNING] 未使用累积结果！原因：")
                    print(f"  - webcam={self.webcam}")
                    print(f"  - has all_labels_dict={hasattr(self, 'all_labels_dict')}")
                    print(f"  - all_labels_dict empty={not self.all_labels_dict if hasattr(self, 'all_labels_dict') else 'N/A'}")
                
                print(f"[DEBUG] results_picture (after) = {self.results_picture}")
                
                # --- 发送图片和表格结果 --- #
                self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                for key, value in self.results_picture.items():
                    self.results_table.append([key, str(value)])
                
                print(f"[DEBUG] results_table = {self.results_table}")
                
                self.results_picture = dict()
                self.send_result_table.emit(self.results_table)  # 发送表格结果
                self.results_table = list()
                # --- 发送图片和表格结果 --- #
                self.all_labels_dict = {}
                self.dataset.running = False  # stop flag for Thread
                # 判断self.dataset里面是否有threads
                if hasattr(self.dataset, 'threads'):
                    for thread in self.dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)  # Add timeout
                if hasattr(self.dataset, 'caps'):
                    for cap in self.dataset.caps:  # Iterate through the stored VideoCapture objects
                        try:
                            cap.release()  # release video capture
                        except Exception as e:
                            LOGGER.warning(f"WARNING Could not release VideoCapture object: {e}")
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break

            #  判断是否更换模型
            if self.current_model_name != self.new_model_name:
                self.send_msg.emit('正在加载模型：{} (Loading Model)'.format(os.path.basename(self.new_model_name)))
                self.setup_model(self.new_model_name)
                self.current_model_name = self.new_model_name
            if self.is_continue:
                if self.is_file:
                    self.send_msg.emit("正在检测文件：{} (Detecting File)".format(os.path.basename(self.source)))
                elif self.webcam and not self.is_url:
                    self.send_msg.emit("正在检测摄像头：Camera_{} (Detecting Webcam)".format(self.source))
                elif self.is_folder:
                    self.send_msg.emit("正在检测文件夹：{} (Detecting Folder)".format(os.path.dirname(self.source[0])))
                elif self.is_url:
                    self.send_msg.emit("正在检测URL：{} (Detecting URL)".format(self.source))
                else:
                    self.send_msg.emit("正在检测：{} (Detecting)".format(self.source))
                self.batch = next(datasets)
                path, im0s, s = self.batch
                self.ori_img = im0s.copy()
                self.vid_cap = self.dataset.cap if self.dataset.mode == "video" else None

                # 使用mediapipe处理图片
                for i, image in enumerate(im0s):
                    black_img = np.zeros(im0s[i].shape, dtype=np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色,opencv默认BGR,mediapipe默认RGB
                    results = self.mp_pose.process(image)
                    if results.pose_landmarks:
                        if self.use_mp:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            mp.solutions.drawing_utils.draw_landmarks(
                                image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                )
                            self.ori_img[i] = image
                        mp.solutions.drawing_utils.draw_landmarks(
                            black_img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                            )
                        im0s[i] = black_img

                # 原始图片送入 input框
                self.send_input.emit(self.ori_img if isinstance(self.ori_img, np.ndarray) else self.ori_img[0])
                count += 1

                # 处理processBar
                if self.vid_cap:
                    if self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                        self.send_progress.emit(percent)
                    else:
                        percent = 100
                        self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                # Preprocess
                with self.dt[0]:
                    im = self.preprocess(im0s)

                # Inference
                with self.dt[1]:
                    preds = self.inference(im)

                # Postprocess
                with self.dt[2]:
                    self.results = self.postprocess(preds, im, im0s)

                n = len(im0s)

                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": self.dt[0].dt * 1e3 / n,
                        "inference": self.dt[1].dt * 1e3 / n,
                        "postprocess": self.dt[2].dt * 1e3 / n,
                    }

                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    self.file_path = p = Path(p)

                    label_str = self.write_results(i, self.results, (p, im, im0))  # labels   /// original :s +=

                    # labels and nums dict
                    class_nums = 0
                    target_nums = 0
                    self.labels_dict = {}
                    if 'no detections' in label_str:
                        pass
                    else:
                        print(f"[DEBUG] ========== 解析检测结果 ==========")
                        print(f"[DEBUG] label_str: {label_str}")
                        for each_target in label_str.split(',')[:-1]:
                            print(f"[DEBUG] 处理目标: {each_target}")
                            num_labelname = list(each_target.split(' '))
                            nums = 0
                            label_name = ""
                            for each in range(len(num_labelname)):
                                if num_labelname[each].isdigit() and each != len(num_labelname) - 1:
                                    nums = num_labelname[each]
                                elif len(num_labelname[each]):
                                    label_name += num_labelname[each] + " "
                            target_nums += int(nums)
                            class_nums += 1
                            print(f"[DEBUG] 解析结果: label_name='{label_name}', nums={nums}")
                            if label_name in self.labels_dict:
                                self.labels_dict[label_name] += int(nums)
                            else:  # 第一次出现的类别
                                self.labels_dict[label_name] = int(nums)
                        print(f"[DEBUG] labels_dict: {self.labels_dict}")
                        print(f"[DEBUG] ======================================")

                    # 累积所有帧的检测结果（摄像头、URL流、视频文件都需要累积）
                    # labels_dict 加入到 all_labels_dict
                    for key, value in self.labels_dict.items():
                        if key in self.all_labels_dict:
                            self.all_labels_dict[key] += value
                        else:
                            self.all_labels_dict[key] = value

                    # 计算累积的类别数和目标数
                    accumulated_class_nums = len(self.all_labels_dict)  # 累积的类别种类数
                    accumulated_target_nums = sum(self.all_labels_dict.values())  # 累积的目标总数

                    self.send_output.emit(self.plotted_img)  # after detection
                    self.send_class_num.emit(accumulated_class_nums)  # 发送累积的类别数
                    self.send_target_num.emit(accumulated_target_nums)  # 发送累积的目标数

                    # 使用累积的结果
                    self.results_picture = self.all_labels_dict

                    if self.save_res:
                        save_path = str(self.save_path / p.name)  # im.jpg
                        self.res_path = self.save_preds(self.vid_cap, i, save_path)

                    if self.speed_thres != 0:
                        time.sleep(self.speed_thres / 1000)  # delay , ms

                if self.is_folder and not is_folder_last:
                    # 判断当前是否为视频
                    if self.file_path and self.file_path.suffix[1:] in VID_FORMATS and percent != self.progress_value:
                        print(f"[DEBUG] 文件夹检测：视频未完成，继续")
                        continue
                    print(f"[DEBUG] 文件夹检测：当前文件检测完成，退出循环（不是最后一个文件）")
                    break

                if percent == self.progress_value and not self.webcam:
                    print(f"[DEBUG] 检测完成，准备发送结果")
                    print(f"[DEBUG] results_picture: {self.results_picture}")
                    self.go_process()
                    self.send_msg.emit('检测完成 (Finish Detection)')
                    # --- 发送图片和表格结果 --- #
                    self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                    for key, value in self.results_picture.items():
                        self.results_table.append([key, str(value)])
                    self.results_picture = dict()
                    self.send_result_table.emit(self.results_table)  # 发送表格结果
                    self.results_table = list()
                    # --- 发送图片和表格结果 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    print(f"[DEBUG] 结果已发送")
                    break

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            weights=model or self.model,
            device=select_device(self.device, verbose=verbose),
            dnn=self.dnn,
            data=self.data,
            fp16=self.half,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.half = self.model.fp16  # update half
        self.model.eval()

        # 添加中文标签映射 - 支持坐姿检测和COCO数据集
        # 尝试从配置文件加载映射，如果失败则使用默认映射
        
        # 坐姿检测类别的默认映射（仅6类实际存在的坐姿）
        chinese_names_map = {
            'normal': '正确坐姿',
            'body_left': '身体左倾',
            'body_right': '身体右倾',
            'left_support_head': '左手托腮',
            'right_support_head': '右手托腮',
            'lying_down': '趴桌',
        }
        
        # 尝试从配置文件加载坐姿检测标签
        try:
            import json
            config_path = os.path.join(os.path.dirname(__file__), '../../config/chinese_labels.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    chinese_names_map.update(config.get('mapping', {}))
        except Exception as e:
            pass  # 如果加载失败，使用默认映射
        
        # 尝试从配置文件加载COCO数据集标签
        try:
            import json
            coco_config_path = os.path.join(os.path.dirname(__file__), '../../config/coco_chinese_labels.json')
            if os.path.exists(coco_config_path):
                with open(coco_config_path, 'r', encoding='utf-8') as f:
                    coco_config = json.load(f)
                    chinese_names_map.update(coco_config.get('mapping', {}))
                    print(f"[DEBUG] 已加载COCO中文标签映射，共{len(coco_config.get('mapping', {}))}个类别")
        except Exception as e:
            print(f"[DEBUG] 加载COCO中文标签失败: {e}")
        
        # 将模型的英文标签替换为中文标签
        if hasattr(self.model, 'names') and self.model.names:
            original_names = self.model.names.copy()
            replaced_count = 0
            
            print(f"[DEBUG] ========== 开始标签映射 ==========")
            print(f"[DEBUG] 原始标签数量: {len(original_names)}")
            print(f"[DEBUG] 映射表大小: {len(chinese_names_map)}")
            
            for key, english_name in original_names.items():
                matched = False
                
                # 首先尝试直接匹配（保留空格和大小写）
                if english_name in chinese_names_map:
                    self.model.names[key] = chinese_names_map[english_name]
                    replaced_count += 1
                    print(f"[DEBUG] 直接匹配: {english_name} -> {chinese_names_map[english_name]}")
                    matched = True
                    continue
                
                # 转换为小写并标准化（去除空格、连字符等）
                english_lower = english_name.lower().strip()
                english_normalized = english_lower.replace(' ', '_').replace('-', '_')
                
                # 尝试标准化后的匹配
                for eng_key, chinese_name in chinese_names_map.items():
                    eng_key_normalized = eng_key.lower().strip().replace(' ', '_').replace('-', '_')
                    
                    # 完全匹配（标准化后）
                    if english_normalized == eng_key_normalized:
                        self.model.names[key] = chinese_name
                        replaced_count += 1
                        print(f"[DEBUG] 标准化匹配: {english_name} -> {chinese_name}")
                        matched = True
                        break
                    
                    # 处理复数形式（英文标签可能是复数，如 kites）
                    if english_normalized.endswith('s') and english_normalized[:-1] == eng_key_normalized:
                        self.model.names[key] = chinese_name
                        replaced_count += 1
                        print(f"[DEBUG] 复数匹配: {english_name} -> {chinese_name}")
                        matched = True
                        break
                    
                    # 反向：配置中的是复数，标签是单数
                    if eng_key_normalized.endswith('s') and eng_key_normalized[:-1] == english_normalized:
                        self.model.names[key] = chinese_name
                        replaced_count += 1
                        print(f"[DEBUG] 反向复数匹配: {english_name} -> {chinese_name}")
                        matched = True
                        break
                
                # 如果没有匹配到，保持原英文名称（不添加任何后缀）
                if not matched:
                    print(f"[DEBUG] 未找到匹配: {english_name} (保持原名)")
            
            print(f"[DEBUG] 标签替换完成: {replaced_count}/{len(original_names)} 个类别已转换为中文")
            print(f"[DEBUG] ========== 替换后的完整标签列表 ==========")
            for key, name in self.model.names.items():
                print(f"[DEBUG]   {key}: {name}")
            print(f"[DEBUG] ==========================================")

        # 加入mediapipe v1.1
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.crop_fraction),
            )
            if self.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.batchsize,
            vid_stride=self.vid_stride,
            buffer=self.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
                self.source_type.stream
                or self.source_type.screenshot
                or len(self.dataset) > 1000  # many images
                or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        return self.model(im, augment=False, visualize=False, embed=False, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
        # Save imgs
        if self.dataset.mode == "image":
            # 使用 cv2.imencode 处理中文路径
            try:
                # 编码图片
                success, encoded_img = cv2.imencode('.jpg', im0)
                if success:
                    # 写入文件（支持中文路径）
                    with open(save_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    print(f"[DEBUG] 图片保存成功: {save_path}")
                else:
                    print(f"[ERROR] 图片编码失败: {save_path}")
            except Exception as e:
                print(f"[ERROR] 保存图片时出错: {e}")
                # 如果失败，尝试使用原始方法
                cv2.imwrite(save_path, im0)
            return save_path

        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                # suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[idx] = cv2.VideoWriter(
                    str(Path(save_path).with_suffix(suffix)), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            # Write video
            self.vid_writer[idx].write(im0)
            return str(Path(save_path).with_suffix(suffix))

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.data_path = p
        result = results[idx]
        log_string += result.verbose()
        result = results[idx]

        result.orig_img = self.ori_img[idx]

        # Add bbox to image
        plot_args = {
            "line_width": self.line_thickness,
            "boxes": True,
            "conf": True,
            "labels": True,
        }
        self.plotted_img = result.plot(**plot_args)
        return log_string
