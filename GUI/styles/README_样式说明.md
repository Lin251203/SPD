# 样式表说明文档

## 📁 文件结构

```
GUI/styles/
├── modern_theme.qss          # 现代化主题样式表（主文件）
└── README_样式说明.md         # 本说明文档
```

---

## 🎨 设计理念

### 核心设计原则

1. **现代化** - 采用渐变色、圆角、阴影等现代UI元素
2. **统一性** - 所有控件遵循统一的设计语言
3. **可读性** - 高对比度，清晰的视觉层次
4. **交互性** - 丰富的悬停、点击反馈效果
5. **专业性** - 适合专业应用场景的配色和布局

### 配色方案

#### 主色调（Primary Colors）
- **紫蓝渐变**: `#667eea` → `#764ba2`
  - 用于：主要按钮、滑块、进度条
  - 寓意：科技感、专业性

- **粉红渐变**: `#f093fb` → `#f5576c`
  - 用于：运行/停止按钮
  - 寓意：活力、动态

- **青蓝渐变**: `#4facfe` → `#00f2fe`
  - 用于：保存按钮
  - 寓意：安全、可靠

#### 辅助色（Secondary Colors）
- **深色文字**: `#2c3e50` - 主要文字颜色
- **白色**: `#ffffff` - 侧边栏文字、按钮文字
- **半透明白**: `rgba(255, 255, 255, 0.1-0.95)` - 背景、边框

#### 状态色（State Colors）
- **悬停**: 渐变反转 + 透明度增加
- **按下**: 颜色加深 + 轻微位移
- **禁用**: 灰色 `#bdc3c7`

---

## 🎯 样式覆盖范围

### 1. 按钮样式（Buttons）

#### 通用按钮
- 紫蓝渐变背景
- 圆角 8px
- 悬停时渐变反转
- 按下时轻微下移

#### 特殊按钮
- **运行/停止按钮**: 粉红渐变
- **保存按钮**: 青蓝渐变
- **侧边栏按钮**: 半透明白色背景
- **窗口控制按钮**: 透明背景，悬停显示

### 2. 输入控件（Input Widgets）

#### SpinBox / DoubleSpinBox
- 白色背景，半透明边框
- 圆角 8px
- 悬停时边框高亮
- 聚焦时边框变为主色

#### ComboBox（下拉框）
- 与 SpinBox 相同风格
- 下拉列表圆角 8px
- 选中项半透明高亮

### 3. 滑块（Sliders）

- 轨道：半透明紫蓝色
- 滑块：紫蓝渐变圆形
- 已滑过部分：紫蓝渐变
- 悬停时滑块放大

### 4. 进度条（Progress Bar）

- 背景：半透明紫蓝色
- 进度：紫蓝渐变
- 圆角 8px
- 居中文字显示

### 5. 标签（Labels）

#### 普通标签
- 白色文字
- 透明背景

#### 状态栏标签
- 悬停时半透明白色背景
- 圆角 5px
- 加粗字体

### 6. 框架（Frames）

#### 显示区域
- 半透明黑色背景
- 紫蓝色边框
- 圆角 15px

#### 状态栏框架
- 半透明紫蓝色背景
- 圆角 12px

### 7. 滚动条（ScrollBars）

- 半透明紫蓝色背景
- 圆角设计
- 悬停时加深

### 8. 工具提示（Tooltips）

- 深色半透明背景
- 紫蓝色边框
- 圆角 8px
- 白色文字

---

## 🔧 使用方法

### 自动加载

样式表会在窗口初始化时自动加载：

```python
class YOLOSHOWWindow(YOLOSHOW):
    def __init__(self):
        super(YOLOSHOWWindow, self).__init__()
        self.load_stylesheet()  # 自动加载样式表
        ...
```

### 手动加载

如果需要手动加载或重新加载样式表：

```python
def load_stylesheet(self):
    """加载统一的样式表"""
    try:
        stylesheet_path = os.path.join(os.getcwd(), 'styles', 'modern_theme.qss')
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, 'r', encoding='utf-8') as f:
                stylesheet = f.read()
                self.setStyleSheet(stylesheet)
    except Exception as e:
        print(f"加载样式表失败: {e}")
```

---

## 🎨 自定义样式

### 修改颜色

如果需要修改配色方案，编辑 `modern_theme.qss` 文件中的颜色值：

```css
/* 修改主色调 */
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #你的颜色1, stop:1 #你的颜色2);
}
```

### 添加新样式

在 `modern_theme.qss` 文件末尾添加新的样式规则：

```css
/* 自定义样式 */
QWidget#my_widget {
    background-color: #your_color;
    border-radius: 10px;
}
```

### 覆盖特定控件样式

使用对象名称（objectName）来覆盖特定控件的样式：

```css
QPushButton#my_special_button {
    background-color: red;
}
```

---

## 📊 样式优先级

1. **内联样式** (最高优先级)
   ```python
   widget.setStyleSheet("background-color: red;")
   ```

2. **对象名称样式**
   ```css
   QPushButton#my_button { ... }
   ```

3. **类样式**
   ```css
   QPushButton { ... }
   ```

4. **通配符样式** (最低优先级)
   ```css
   * { ... }
   ```

---

## 🔍 调试技巧

### 查看当前样式

```python
# 获取控件的当前样式
current_style = widget.styleSheet()
print(current_style)
```

### 临时测试样式

```python
# 临时应用样式进行测试
widget.setStyleSheet("background-color: red; border: 2px solid blue;")
```

### 重新加载样式表

```python
# 修改样式表后重新加载
self.load_stylesheet()
```

---

## 🎯 最佳实践

### 1. 保持一致性

- 使用统一的圆角大小（8px, 10px, 12px, 15px）
- 使用统一的间距（5px, 8px, 10px, 12px, 15px）
- 使用统一的字体大小（9pt, 10pt, 11pt, 12pt）

### 2. 性能优化

- 避免过度使用渐变和阴影
- 合理使用透明度
- 避免复杂的选择器

### 3. 可维护性

- 使用注释分隔不同的样式区域
- 使用有意义的对象名称
- 保持样式表的组织结构清晰

### 4. 响应式设计

- 考虑不同屏幕尺寸
- 使用相对单位（em, %, 等）
- 测试不同分辨率下的显示效果

---

## 🌈 配色参考

### 渐变色生成工具

- [uiGradients](https://uigradients.com/)
- [WebGradients](https://webgradients.com/)
- [Gradient Hunt](https://gradienthunt.com/)

### 配色方案工具

- [Coolors](https://coolors.co/)
- [Adobe Color](https://color.adobe.com/)
- [Material Design Colors](https://materialui.co/colors)

---

## 📝 更新日志

### v1.0.0 (2024-12-16)

**新增功能：**
- ✅ 创建统一的现代化主题样式表
- ✅ 支持所有主要控件的样式定制
- ✅ 添加悬停、点击等交互效果
- ✅ 统一配色方案和设计语言
- ✅ 添加工具提示样式
- ✅ 添加滚动条样式
- ✅ 支持响应式设计

**设计特点：**
- 🎨 紫蓝渐变主色调
- 🔵 圆角设计
- ✨ 半透明效果
- 🎯 高对比度
- 💫 平滑过渡动画

---

## 🤝 贡献指南

如果您想改进样式表，请遵循以下步骤：

1. 在 `modern_theme.qss` 中进行修改
2. 测试所有相关控件的显示效果
3. 确保与整体设计风格一致
4. 更新本文档的相关说明
5. 提交更改并说明修改原因

---

## 📞 技术支持

如果遇到样式相关问题：

1. 检查样式表文件路径是否正确
2. 检查样式表语法是否正确
3. 检查控件的对象名称是否匹配
4. 查看控制台是否有错误信息
5. 尝试重新加载样式表

---

## 🎓 学习资源

### Qt 样式表文档

- [Qt Style Sheets Reference](https://doc.qt.io/qt-6/stylesheet-reference.html)
- [Qt Style Sheets Examples](https://doc.qt.io/qt-6/stylesheet-examples.html)
- [Qt Style Sheets Syntax](https://doc.qt.io/qt-6/stylesheet-syntax.html)

### CSS 基础

- [MDN CSS 文档](https://developer.mozilla.org/zh-CN/docs/Web/CSS)
- [CSS Tricks](https://css-tricks.com/)

---

## 📄 许可证

本样式表遵循项目的整体许可证。

---

**最后更新**: 2024年12月16日
**版本**: 1.0.0
**作者**: Kiro AI Assistant
