from utils import glo
import traceback


def yoloshowvsSHOW():
    try:
        print("[DEBUG] 开始显示对比模式窗口")
        yoloshow_glo = glo.get_value('yoloshow')
        yoloshowvs_glo = glo.get_value('yoloshowvs')
        
        print("[DEBUG] 设置模型名称")
        glo.set_value('yoloname1', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 yolov5-seg yolov8-seg yolov11-seg rtdetr yolov8-pose yolov11-pose yolov8-obb yolov11-obb")
        glo.set_value('yoloname2', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 yolov5-seg yolov8-seg yolov11-seg rtdetr yolov8-pose yolov11-pose yolov8-obb yolov11-obb")
        
        # 注释掉 reloadModel()，因为它会导致线程对象引用的类定义失效
        # 在窗口切换时不需要重新加载模块，所有模型类型已经在应用启动时加载
        # print("[DEBUG] 重新加载模型")
        # yoloshowvs_glo.reloadModel()
        
        print("[DEBUG] 显示对比模式窗口")
        yoloshowvs_glo.show()
        
        print("[DEBUG] 清理单模式窗口")
        if hasattr(yoloshow_glo, 'animation_window'):
            yoloshow_glo.animation_window = None
        if hasattr(yoloshow_glo, 'closed'):
            try:
                yoloshow_glo.closed.disconnect()
            except:
                pass  # 如果没有连接，忽略错误
        
        print("[DEBUG] 对比模式窗口显示成功")
    except Exception as e:
        print(f"[ERROR] 显示对比模式窗口失败: {e}")
        traceback.print_exc()


def yoloshowSHOW():
    try:
        print("[DEBUG] 开始显示单模式窗口")
        yoloshow_glo = glo.get_value('yoloshow')
        yoloshowvs_glo = glo.get_value('yoloshowvs')
        
        print("[DEBUG] 设置模型名称")
        glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov11 yolov5-seg yolov8-seg yolov11-seg rtdetr yolov8-pose yolov11-pose yolov8-obb yolov11-obb")
        
        # 注释掉 reloadModel()，因为它会导致线程对象引用的类定义失效
        # 在窗口切换时不需要重新加载模块，所有模型类型已经在应用启动时加载
        # print("[DEBUG] 重新加载模型")
        # yoloshow_glo.reloadModel()
        
        print("[DEBUG] 显示单模式窗口")
        yoloshow_glo.show()
        
        print("[DEBUG] 清理对比模式窗口")
        if hasattr(yoloshowvs_glo, 'animation_window'):
            yoloshowvs_glo.animation_window = None
        if hasattr(yoloshowvs_glo, 'closed'):
            try:
                yoloshowvs_glo.closed.disconnect()
            except:
                pass  # 如果没有连接，忽略错误
        
        print("[DEBUG] 单模式窗口显示成功")
    except Exception as e:
        print(f"[ERROR] 显示单模式窗口失败: {e}")
        traceback.print_exc()


def yoloshow2vs():
    try:
        print("[DEBUG] 切换到对比模式")
        yoloshow_glo = glo.get_value('yoloshow')
        yoloshow_glo.closed.connect(yoloshowvsSHOW)
        yoloshow_glo.close()
        print("[DEBUG] 单模式窗口已关闭")
    except Exception as e:
        print(f"[ERROR] 切换到对比模式失败: {e}")
        traceback.print_exc()


def vs2yoloshow():
    try:
        print("[DEBUG] 切换到单模式")
        yoloshowvs_glo = glo.get_value('yoloshowvs')
        yoloshowvs_glo.closed.connect(yoloshowSHOW)
        yoloshowvs_glo.close()
        print("[DEBUG] 对比模式窗口已关闭")
    except Exception as e:
        print(f"[ERROR] 切换到单模式失败: {e}")
        traceback.print_exc()