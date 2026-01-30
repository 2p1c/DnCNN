# CNN Deep Learning Project Rules

## 🧠 1. 核心协作原则 (Interaction Protocol)

### 🛑 理解优先 (Understanding First)
在编写任何代码之前，必须确保完全理解用户的意图。**不要猜测**。

- **分析与确认**: 当收到修改或生成代码的指令时，首先分析所需信息。
- **主动提问**: 如果信息不足，礼貌地要求澄清。
- **思维模拟**: 在决策时，模拟领域内"最强大脑"（如 Andrej Karpathy 或 Yann LeCun）的思维方式，思考如果是他们会如何构建这个模型。

### 💬 沟通标准范例
当用户请求 "写一个 CNN" 时，**必须**按以下格式回应，而不是直接写代码：

> "我理解您想实现一个 CNN 模型。为了构建最专业的架构，我需要确认以下细节：
> 1. **框架偏好**: PyTorch, TensorFlow, 还是 JAX?
> 2. **具体任务**: 图像分类、目标检测、还是语义分割?
> 3. **数据规格**: 输入图像的维度 (C, H, W) 是多少?
> 4. **架构约束**: 是否有特定的 backbone 要求 (如 ResNet, EfficientNet) 或算力限制?
>
> 请确认以上信息，我将为您设计最佳方案。"

---

## 💻 2. 代码质量标准 (Code Quality Standards)

### 📝 通用原则
- **可读性第一**: 清晰胜过简洁。避免过度复杂的单行代码。
- **模块化设计**: 模型定义 (`model.py`)、数据加载 (`dataset.py`) 和 训练循环 (`train.py`) 必须解耦。
- **错误处理**: 对文件读取、GPU/CPU 设备切换、维度不匹配等常见错误进行 `try-except` 处理或 `assert` 检查。

### 🐍 Python & Deep Learning 规范

1. **类型标注 (Type Hinting)**:
   - 所有函数参数必须包含类型标注。
   - 对于 Tensor，尽可能注明预期形状。
   ```python
   # Good
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       # x shape: [batch_size, channels, height, width]
       ...
   ```

2. **文档注释 (Documentation)**:
   - 使用 Google Style 或 NumPy Style 的 Docstring。
   - 必须解释 "为什么" (Why) 选择了这个层或参数，而不仅仅是 "做了什么"。

3. **张量形状注释 (Shape Comments)**:
   - 在卷积层、池化层或 View/Reshape 操作后，必须添加注释说明张量的形状变化。
   ```python
   x = self.conv1(x)  # [32, 64, 128, 128] -> [32, 128, 64, 64]
   ```

4. **可复现性 (Reproducibility)**:
   - 代码必须包含设置随机种子的工具函数（Seed everything）。

---

## 🔬 3. 深度学习最佳实践 (Deep Learning Best Practices)

### 训练前检查
- **数据检查**: 在开始训练前，编写一个小脚本可视化一个 batch 的数据，确保预处理（Normalize/Augmentation）正确。
- **Overfit First**: 在完整数据集训练前，建议先使用一个小样本（如 10 张图）进行训练，确保 Loss 能迅速下降到 0，以验证模型实现的正确性。

### 配置管理
- **配置分离**: 超参数（Learning rate, Batch size, Channels）不应硬编码在模型文件中，应通过 config 对象或 argparse 传入。
