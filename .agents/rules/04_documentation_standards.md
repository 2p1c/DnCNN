# Documentation & Visualization

## 📊 Logic Visualization
当你编写复杂的模型架构、数据流处理或状态机逻辑时，**主动**在回复中包含一个 Mermaid 流程图。

示例：
```mermaid
graph TD
    A[Input Image] --> B[Backbone (ResNet)]
    B --> C[Feature Map]
    C --> D[RPN Head]
    C --> E[ROI Align]
```

## 📝 Docstring Standards
- **Args & Returns**: 所有的 Docstring 必须清晰列出参数形状 (Shapes) 和返回类型。
- **Usage Example**: 对于核心工具函数，在 Docstring 中包含一个简短的 Example 用法示例。

## 📦 README Updates
- 如果你添加了新的依赖库 (libraries)，请提醒我更新 `requirements.txt` 或 `environment.yml`。
- 如果你添加了新的核心脚本，请提供一段简短的描述，用于更新项目的 `README.md`。
