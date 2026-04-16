"""
向量库字段设计和模式定义
为健身知识库设计优化的向量存储结构
"""

from typing import Dict, Any, List, Optional
from pymilvus import MilvusClient
from loguru import logger


class VectorSchema:
    """
    向量库模式定义

    为健身知识库设计的高效向量存储结构，支持：
    - 多级元数据索引
    - 快速过滤查询
    - 高效的向量检索
    """

    # Qwen3-Embedding-0.6B 的向量维度
    EMBEDDING_DIMENSION = 1536

    # 推荐的集合名称
    COLLECTION_NAMES = {
        "fitness_knowledge": "fitness_knowledge",      # 健身知识库
        "exercise_guide": "exercise_guide",            # 运动指导
        "nutrition_science": "nutrition_science",      # 营养科学
        "injury_prevention": "injury_prevention",      # 损伤预防
    }

    @staticmethod
    def get_fitness_knowledge_schema() -> Dict[str, Any]:
        """
        获取健身知识库的集合模式

        Returns:
            集合模式定义
        """
        schema = MilvusClient.create_schema(
            auto_id=False,  # 使用自定义ID，便于管理
            enable_dynamic_field=True,  # 允许动态字段
            fields=[
                # 主键字段
                {
                    "field_name": "id",
                    "datatype": "VARCHAR",
                    "is_primary": True,
                    "max_length": 256,
                    "description": "文档唯一标识符",
                },
                # 向量字段
                {
                    "field_name": "embedding",
                    "datatype": "FLOAT_VECTOR",
                    "dim": VectorSchema.EMBEDDING_DIMENSION,
                    "description": "Qwen3嵌入向量",
                },
                # 内容字段
                {
                    "field_name": "content",
                    "datatype": "VARCHAR",
                    "max_length": 65535,
                    "description": "文档内容",
                },
                # 书籍信息字段
                {
                    "field_name": "book_title",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "书籍标题",
                },
                {
                    "field_name": "book_authors",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "书籍作者",
                },
                {
                    "field_name": "book_publisher",
                    "datatype": "VARCHAR",
                    "max_length": 256,
                    "description": "出版社",
                },
                # 章节信息字段
                {
                    "field_name": "chapter_title",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "章节标题",
                },
                {
                    "field_name": "section_title",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "小节标题",
                },
                {
                    "field_name": "subsection_title",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "子小节标题",
                },
                # 分类字段
                {
                    "field_name": "category",
                    "datatype": "VARCHAR",
                    "max_length": 128,
                    "description": "知识类别 (训练学/营养学/生理学/解剖学/损伤预防)",
                },
                {
                    "field_name": "subcategory",
                    "datatype": "VARCHAR",
                    "max_length": 128,
                    "description": "子类别 (如: 力量训练/有氧训练/柔韧性训练)",
                },
                {
                    "field_name": "difficulty_level",
                    "datatype": "VARCHAR",
                    "max_length": 32,
                    "description": "难度等级 (初级/中级/高级)",
                },
                # 内容特征字段
                {
                    "field_name": "content_type",
                    "datatype": "VARCHAR",
                    "max_length": 64,
                    "description": "内容类型 (理论/实践/案例分析/FAQ)",
                },
                {
                    "field_name": "target_audience",
                    "datatype": "VARCHAR",
                    "max_length": 128,
                    "description": "目标读者 (初学者/中级者/高级者/教练员/研究者)",
                },
                # 质量控制字段
                {
                    "field_name": "content_length",
                    "datatype": "INT",
                    "description": "内容长度（字符数）",
                },
                {
                    "field_name": "quality_score",
                    "datatype": "FLOAT",
                    "description": "质量分数 (0-1)",
                },
                # 来源信息字段
                {
                    "field_name": "source_file",
                    "datatype": "VARCHAR",
                    "max_length": 512,
                    "description": "源文件路径",
                },
                {
                    "field_name": "source_type",
                    "datatype": "VARCHAR",
                    "max_length": 64,
                    "description": "来源类型 (书籍/论文/网站/视频)",
                },
                {
                    "field_name": "extract_date",
                    "datatype": "VARCHAR",
                    "max_length": 32,
                    "description": "提取日期",
                },
                # 统计字段
                {
                    "field_name": "chunk_count",
                    "datatype": "INT",
                    "description": "包含的原始块数",
                },
                {
                    "field_name": "view_count",
                    "datatype": "INT",
                    "description": "查看次数",
                },
                # 索引字段
                {
                    "field_name": "keywords",
                    "datatype": "ARRAY<VARCHAR(128)>",
                    "description": "关键词列表",
                },
                {
                    "field_name": "tags",
                    "datatype": "ARRAY<VARCHAR(64)>",
                    "description": "标签列表",
                },
            ],
            description="健身知识向量库",
        )

        return schema

    @staticmethod
    def get_index_config() -> Dict[str, Any]:
        """
        获取索引配置

        Returns:
            索引配置字典
        """
        # 向量索引配置
        vector_index_config = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",  # 内积，适合归一化向量
            "params": {"nlist": 128},  # 聚类中心数
        }

        # 标量字段索引配置
        scalar_indexes = [
            # 书名索引
            {
                "field_name": "book_title",
                "index_type": "INVERTED",
                "index_name": "book_title_idx",
            },
            # 类别索引
            {
                "field_name": "category",
                "index_type": "INVERTED",
                "index_name": "category_idx",
            },
            # 难度等级索引
            {
                "field_name": "difficulty_level",
                "index_type": "INVERTED",
                "index_name": "difficulty_level_idx",
            },
            # 质量分数索引
            {
                "field_name": "quality_score",
                "index_type": "STL_SORT",  # 排序索引
                "index_name": "quality_score_idx",
            },
            # 内容长度索引
            {
                "field_name": "content_length",
                "index_type": "STL_SORT",
                "index_name": "content_length_idx",
            },
        ]

        return {
            "vector_index": vector_index_config,
            "scalar_indexes": scalar_indexes,
        }

    @staticmethod
    def create_collection(
        client: MilvusClient,
        collection_name: str = "fitness_knowledge",
        drop_existing: bool = False,
    ) -> bool:
        """
        创建向量库集合

        Args:
            client: Milvus客户端
            collection_name: 集合名称
            drop_existing: 是否删除已存在的集合

        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            if client.has_collection(collection_name):
                if drop_existing:
                    logger.info(f"删除已存在的集合: {collection_name}")
                    client.drop_collection(collection_name)
                else:
                    logger.info(f"集合已存在: {collection_name}")
                    return True

            # 获取模式
            schema = VectorSchema.get_fitness_knowledge_schema()

            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

            logger.success(f"✅ 集合创建成功: {collection_name}")

            # 创建向量索引
            index_config = VectorSchema.get_index_config()
            vector_index = index_config["vector_index"]

            client.create_index(
                collection_name=collection_name,
                field_name="embedding",
                index_config=vector_index,
            )

            logger.info(f"✅ 向量索引创建成功")

            # 创建标量索引
            for scalar_index in index_config["scalar_indexes"]:
                try:
                    client.create_index(
                        collection_name=collection_name,
                        field_name=scalar_index["field_name"],
                        index_config={
                            "index_type": scalar_index["index_type"],
                        },
                    )
                    logger.info(f"✅ 标量索引创建成功: {scalar_index['field_name']}")
                except Exception as e:
                    logger.warning(f"标量索引创建失败: {scalar_index['field_name']} - {e}")

            # 加载集合到内存
            client.load_collection(collection_name)
            logger.info(f"✅ 集合已加载到内存")

            return True

        except Exception as e:
            logger.error(f"集合创建失败: {e}")
            return False


class VectorDocumentBuilder:
    """
    向量文档构建器

    将处理后的文档块转换为向量库文档格式
    """

    @staticmethod
    def build_document(
        chunk: Dict[str, Any],
        embedding: List[float],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        构建向量库文档

        Args:
            chunk: 文档块
            embedding: 嵌入向量
            document_id: 文档ID

        Returns:
            向量库文档
        """
        content = chunk["content"]
        metadata = chunk.get("metadata", {})

        # 提取分类信息（可以基于内容分析）
        category = VectorDocumentBuilder._infer_category(content, metadata)
        difficulty = VectorDocumentBuilder._infer_difficulty(content)

        # 构建文档
        document = {
            "id": document_id,
            "embedding": embedding,
            "content": content,

            # 书籍信息
            "book_title": metadata.get("book_title", ""),
            "book_authors": metadata.get("authors", ""),
            "book_publisher": metadata.get("publisher", ""),

            # 章节信息
            "chapter_title": metadata.get("chapter_title", ""),
            "section_title": metadata.get("section_title", ""),
            "subsection_title": metadata.get("subsection_title", ""),

            # 分类信息
            "category": category,
            "subcategory": metadata.get("subcategory", ""),
            "difficulty_level": difficulty,

            # 内容特征
            "content_type": metadata.get("content_type", "理论"),
            "target_audience": metadata.get("target_audience", "通用"),

            # 质量控制
            "content_length": len(content),
            "quality_score": VectorDocumentBuilder._calculate_quality_score(content),

            # 来源信息
            "source_file": metadata.get("source_file", ""),
            "source_type": "书籍",
            "extract_date": metadata.get("extract_date", ""),

            # 统计信息
            "chunk_count": metadata.get("chunk_count", 1),
            "view_count": 0,

            # 索引字段
            "keywords": VectorDocumentBuilder._extract_keywords(content),
            "tags": VectorDocumentBuilder._generate_tags(content, category),
        }

        return document

    @staticmethod
    def _infer_category(content: str, metadata: Dict[str, Any]) -> str:
        """
        推断内容类别

        Args:
            content: 文档内容
            metadata: 元数据

        Returns:
            类别字符串
        """
        content_lower = content.lower()

        # 基于关键词推断类别
        category_keywords = {
            "训练学": ["训练", "练习", "肌肉", "力量", "耐力", "爆发力", "速度"],
            "营养学": ["营养", "蛋白质", "碳水化合物", "脂肪", "维生素", "矿物质", "饮食"],
            "生理学": ["生理", "代谢", "内分泌", "神经", "心血管", "呼吸"],
            "解剖学": ["解剖", "肌肉", "骨骼", "关节", "韧带", "肌腱"],
            "损伤预防": ["损伤", "预防", "康复", "保护", "安全"],
        }

        # 统计关键词出现频率
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                category_scores[category] = score

        # 返回得分最高的类别
        if category_scores:
            return max(category_scores, key=category_scores.get)

        return "通用"

    @staticmethod
    def _infer_difficulty(content: str) -> str:
        """
        推断内容难度

        Args:
            content: 文档内容

        Returns:
            难度等级
        """
        # 简单的启发式规则
        if len(content) < 200:
            return "初级"
        elif len(content) < 500:
            return "中级"
        else:
            return "高级"

    @staticmethod
    def _calculate_quality_score(content: str) -> float:
        """
        计算内容质量分数

        Args:
            content: 文档内容

        Returns:
            质量分数 (0-1)
        """
        score = 0.5  # 基础分数

        # 长度适中的内容得分更高
        length = len(content)
        if 100 <= length <= 1000:
            score += 0.2
        elif length > 1000:
            score += 0.1

        # 包含专业术语的内容得分更高
        professional_terms = ["机制", "原理", "研究", "实验", "数据"]
        term_count = sum(1 for term in professional_terms if term in content)
        score += min(term_count * 0.1, 0.3)

        return min(score, 1.0)

    @staticmethod
    def _extract_keywords(content: str) -> List[str]:
        """
        提取关键词

        Args:
            content: 文档内容

        Returns:
            关键词列表
        """
        # 简单的关键词提取（可以替换为更复杂的算法）
        keywords = []

        # 常见的健身相关关键词
        fitness_keywords = [
            "肌肉", "力量", "训练", "营养", "恢复", "脂肪", "蛋白质",
            "有氧", "无氧", "耐力", "爆发力", "柔韧性", "平衡",
        ]

        for keyword in fitness_keywords:
            if keyword in content:
                keywords.append(keyword)

        return keywords[:5]  # 最多返回5个关键词

    @staticmethod
    def _generate_tags(content: str, category: str) -> List[str]:
        """
        生成标签

        Args:
            content: 文档内容
            category: 内容类别

        Returns:
            标签列表
        """
        tags = [category]

        # 基于内容添加额外标签
        if "实践" in content or "方法" in content:
            tags.append("实用")
        if "研究" in content or "实验" in content:
            tags.append("科学")
        if "初级" in content or "入门" in content:
            tags.append("入门")

        return list(set(tags))  # 去重


# 使用示例
def example_usage():
    """
    向量库模式使用示例
    """
    print("=" * 60)
    print("向量库模式设计示例")
    print("=" * 60)

    # 显示模式信息
    print("\n1. 向量维度:")
    print(f"  {VectorSchema.EMBEDDING_DIMENSION}")

    print("\n2. 推荐的集合名称:")
    for name, collection in VectorSchema.COLLECTION_NAMES.items():
        print(f"  {name}: {collection}")

    print("\n3. 索引配置:")
    index_config = VectorSchema.get_index_config()
    print(f"  向量索引类型: {index_config['vector_index']['index_type']}")
    print(f"  距离度量: {index_config['vector_index']['metric_type']}")
    print(f"  标量索引数量: {len(index_config['scalar_indexes'])}")

    # 示例文档构建
    print("\n4. 示例文档构建:")
    sample_chunk = {
        "content": "卧推是锻炼胸肌的最佳动作，主要包括杠铃卧推和哑铃卧推。这项运动可以有效刺激胸大肌、三角肌前束和肱三头肌。",
        "metadata": {
            "book_title": "力量训练原理与实践",
            "chapter_title": "上肢训练",
            "source_file": "/path/to/book.md",
        }
    }

    # 模拟嵌入向量
    import random
    fake_embedding = [random.random() for _ in range(VectorSchema.EMBEDDING_DIMENSION)]

    document = VectorDocumentBuilder.build_document(
        sample_chunk,
        fake_embedding,
        "doc_001"
    )

    print(f"  文档ID: {document['id']}")
    print(f"  类别: {document['category']}")
    print(f"  难度: {document['difficulty_level']}")
    print(f"  质量分数: {document['quality_score']:.2f}")
    print(f"  关键词: {document['keywords']}")
    print(f"  标签: {document['tags']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()