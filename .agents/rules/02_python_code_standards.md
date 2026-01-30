# Python Senior Engineer Standards

## 🐍 Modern Python Idioms
- **Pathlib over OS**: 始终使用 `pathlib.Path` 进行文件路径操作，禁止使用 `os.path`。
- **Type Hints**: 即使是简单的脚本，也必须使用 `typing` (List, Dict, Optional, Union) 或 Python 3.9+ 的原生类型提示。
- **Pydantic**: 在处理复杂数据结构或配置时，优先使用 `Pydantic` 模型而不是纯字典。
- **F-strings**: 始终使用 f-strings 进行字符串格式化。

## 🛡️ Defensive Programming
- **Early Returns**: 优先使用"卫语句"（Guard Clauses）来减少嵌套层级。
- **Explicit Imports**: 避免 `from module import *`，必须显式导入使用的函数或类。
- **Logging**: 在生产级代码中，使用 `logging` 模块而不是 `print`。

## ⚡ Performance Awareness
- 在处理大型数组/列表时，优先使用生成器 (Generators) 或 `itertools`。
- 涉及数值计算时，必须向量化 (Vectorization) 操作 (NumPy/PyTorch)，禁止使用 Python 原生 `for` 循环处理数据。
