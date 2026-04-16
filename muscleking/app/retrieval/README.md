# Qwen3 模型使用说明

## 模型介绍

本项目使用 Qwen 系列的嵌入和重排序模型：

- **Qwen3-Embedding-0.6B**: 嵌入模型，用于将文本转换为向量表示
- **Qwen3-Reranker-0.6B**: 重排序模型，用于对检索结果进行精细排序

## 模型位置

```
muscleking/app/retrieval/
├── embedding_model/
│   └── qwen_embedding.py       # 嵌入模型
└── reranker/
    └── qwen_reranker.py        # 重排序模型
```

## 安装依赖

确保安装了以下依赖：

```bash
# 使用 uv 安装
uv add sentence-transformers torch scikit-learn

# 或使用 pip
pip install sentence-transformers torch scikit-learn
```

## 快速开始

### 1. 测试模型（首次运行会下载模型）

```bash
cd /home/wenkai/project/MuslceKing
python -m muscleking.app.scripts.test_qwen_models
```

首次运行时，模型会自动从 HuggingFace 下载到本地缓存：
- Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

### 2. 在代码中使用

#### 嵌入模型

```python
from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding

# 初始化模型
embedding_model = QwenEmbedding()

# 编码单个文本
text = "卧推是锻炼胸肌的最佳动作"
embedding = embedding_model.encode_single(text)

# 批量编码
texts = ["文本1", "文本2", "文本3"]
embeddings = embedding_model.encode(texts, batch_size=32)

# 获取模型信息
info = embedding_model.get_model_info()
print(f"嵌入维度: {embedding_model.get_dimension()}")
```

#### 重排序模型

```python
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker

# 初始化模型
reranker = QwenReranker()

# 准备数据
query = "如何锻炼胸肌"
documents = [
    {
        "content": "卧推是锻炼胸肌的最佳动作",
        "source": "fitness_guide"
    },
    {
        "content": "深蹲主要锻炼腿部肌肉",
        "source": "leg_training"
    }
]

# 重排序
reranked_docs = reranker.rerank(query, documents, top_k=5)

# 查看结果
for doc in reranked_docs:
    score = doc.get("rerank_score", 0.0)
    content = doc.get("content", "")
    print(f"[{score:.4f}] {content}")

# 根据阈值过滤
filtered_docs = reranker.rerank_by_threshold(
    query, documents, threshold=0.5
)
```

## 配置选项

### 嵌入模型配置

```python
embedding_model = QwenEmbedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",  # 模型名称
    device=None,                              # 自动检测设备 (cuda/cpu)
    cache_folder=None,                        # 自定义缓存目录
)
```

### 重排序模型配置

```python
reranker = QwenReranker(
    model_name="Qwen/Qwen3-Reranker-0.6B",   # 模型名称
    device=None,                              # 自动检测设备 (cuda/cpu)
    cache_folder=None,                        # 自定义缓存缓存目录
)
```

## 在 RAG 系统中集成

### 与知识库检索集成

```python
from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker

class EnhancedRAGSystem:
    def __init__(self):
        self.embedding_model = QwenEmbedding()
        self.reranker = QwenReranker()
    
    async def search(self, query: str, top_k: int = 10):
        # 1. 向量检索
        query_embedding = self.embedding_model.encode_single(query)
        candidates = await self.vector_search(query_embedding, top_k=20)
        
        # 2. 重排序
        reranked_docs = self.reranker.rerank(query, candidates, top_k=top_k)
        
        return reranked_docs
```

## 性能优化建议

1. **批处理**: 对于大量文本，使用批处理可以提高效率
   ```python
   embeddings = embedding_model.encode(texts, batch_size=32)
   ```

2. **GPU加速**: 如果有 NVIDIA GPU，模型会自动使用 CUDA
   ```python
   # 检查设备
   import torch
   print(torch.cuda.is_available())  # 应该返回 True
   ```

3. **缓存管理**: 可以指定自定义缓存目录
   ```python
   embedding_model = QwenEmbedding(
       cache_folder="/path/to/cache"
   )
   ```

## 模型规格

| 模型 | 参数量 | 嵌入维度 | 最大序列长度 |
|------|--------|----------|--------------|
| Qwen3-Embedding-0.6B | 0.6B | 1536 | 32768 |
| Qwen3-Reranker-0.6B | 0.6B | - | 32768 |

## 故障排除

### 1. 模型下载慢

可以使用镜像站加速下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python test_qwen_models.py
```

### 2. 内存不足

如果遇到内存问题，可以：
- 使用 CPU 模式：`device="cpu"`
- 减小批处理大小：`batch_size=8`

### 3. 模型版本更新

如果需要更新到最新版本：

```python
# 清除缓存重新下载
import shutil
shutil.rmtree("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B")
```

## 参考资源

- [Qwen 官方文档](https://qwen.readthedocs.io/)
- [Sentence-Transformers 文档](https://www.sbert.net/)
- [HuggingFace 模型页面](https://huggingface.co/Qwen)