"""
WGER 数据导入脚本

从 https://github.com/wger-project/wger 获取健身数据，构建知识图谱。

WGER 提供的 API:
- https://wger.de/api/v2/exerciseinfo/?limit=100&offset=0
- https://wger.de/api/v2/muscle/
- https://wger.de/api/v2/equipment/
- https://wger.de/api/v2/exercisecategory/

使用方法:
    python scripts/wger_ingest.py [--limit N] [--batch-size N] [--clear]
"""

import asyncio
import argparse
import re
import aiohttp
from typing import Any, Dict, List, Optional
from pathlib import Path
from loguru import logger
import sys

# 添加项目根目录到路径
# 脚本在 muscleking/scripts/ 下，需要向上3级到达项目根目录 MuscleKing/
scripts_dir = Path(__file__).resolve().parent
scripts_parent = scripts_dir.parent  # muscleking/
project_root = scripts_parent.parent  # MuscleKing/

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_parent))

# Neo4j 连接配置 - 如果 settings 不可用则使用默认值
try:
    from muscleking.app.config.settings import settings

    NEO4J_URI = getattr(settings, "NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = getattr(settings, "NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = getattr(settings, "NEO4J_PASSWORD", "muscleking")
    NEO4J_DATABASE = getattr(settings, "NEO4J_DATABASE", "neo4j")
except ImportError:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "muscleking"
    NEO4J_DATABASE = "neo4j"

from muscleking.app.storage.core.neo4jconn import get_neo4j_graph


# WGER API 基础 URL
WGER_API_BASE = "https://wger.de/api/v2"
WGER_API_EXERCISES = f"{WGER_API_BASE}/exerciseinfo"
WGER_API_MUSCLES = f"{WGER_API_BASE}/muscle"
WGER_API_EQUIPMENT = f"{WGER_API_BASE}/equipment"
WGER_API_CATEGORIES = f"{WGER_API_BASE}/exercisecategory"


class WGERIngester:
    """WGER 数据导入器"""

    def __init__(
        self,
        neo4j_graph,
        limit: Optional[int] = None,
        batch_size: int = 50,
    ):
        self.neo4j_graph = neo4j_graph
        self.limit = limit
        self.batch_size = batch_size

    async def fetch_all_data(self) -> Dict[str, List[Dict]]:
        """获取 WGER 的所有基础数据"""
        logger.info("正在从 WGER 获取数据...")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_data(session, WGER_API_MUSCLES, "muscles"),
                self._fetch_data(session, WGER_API_EQUIPMENT, "equipment"),
                self._fetch_data(session, WGER_API_CATEGORIES, "categories"),
                self._fetch_exercises(session),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            data = {
                "muscles": [],
                "equipment": [],
                "categories": [],
                "exercises": [],
            }

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"获取数据失败: {result}")
                else:
                    key = list(data.keys())[i]
                    data[key] = result

            total = len(data["exercises"])
            logger.info(
                f"获取到 {total} 个训练动作, {len(data['muscles'])} 个肌群, {len(data['equipment'])} 个器械"
            )

            return data

    async def _fetch_data(
        self, session: aiohttp.ClientSession, url: str, key: str
    ) -> List[Dict[str, Any]]:
        """获取单个 API 端点的数据"""
        try:
            params = {"limit": 500}  # 获取尽可能多的数据
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"API 请求失败 {url}: {response.status}")
                    return []

                data = await response.json()
                return data.get("results", [])
        except Exception as e:
            logger.error(f"获取 {key} 数据失败: {e}")
            return []

    async def _fetch_exercises(
        self, session: aiohttp.ClientSession
    ) -> List[Dict[str, Any]]:
        """获取所有练习数据"""
        all_exercises = []
        offset = 0

        while True:
            try:
                params = {"limit": 100, "offset": offset}
                async with session.get(WGER_API_EXERCISES, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"获取练习数据失败: {response.status}")
                        break

                    data = await response.json()
                    exercises = data.get("results", [])

                    if not exercises:
                        break

                    all_exercises.extend(exercises)
                    logger.info(f"已获取 {len(all_exercises)} 个练习...")

                    if self.limit and len(all_exercises) >= self.limit:
                        all_exercises = all_exercises[: self.limit]
                        break

                    if len(exercises) < 100:
                        break

                    offset += 100

            except Exception as e:
                logger.error(f"获取练习数据失败: {e}")
                break

        return all_exercises

    def clean_html(self, html: str) -> str:
        """清除 HTML 标签，提取纯文本"""
        if not html:
            return ""

        # 移除 HTML 标签
        text = re.sub(r"<[^>]+>", "", html)
        # 替换 HTML 实体
        text = text.replace("&nbsp;", " ")
        text = text.replace("&", "&")
        text = text.replace("<", "<")
        text = text.replace(">>")
        text = text.replace('"', '"')
        # 清理多余空白
        text = re.sub(r"\s+", " ", text).strip()

        return text

    async def create_graph_nodes_and_relationships(self, data: Dict[str, List[Dict]]):
        """构建知识图谱"""
        logger.info("开始构建知识图谱...")

        # 先创建基础数据
        await self._create_muscles(data.get("muscles", []))
        await self._create_equipment(data.get("equipment", []))
        await self._create_categories(data.get("categories", []))

        # 创建练习及其关系
        exercises = data.get("exercises", [])
        total = len(exercises)

        for i, exercise in enumerate(exercises):
            try:
                await self._create_exercise(exercise)

                if (i + 1) % self.batch_size == 0:
                    logger.info(f"进度: {i + 1}/{total} ({100 * (i + 1) // total}%)")

            except Exception as e:
                logger.warning(
                    f"创建练习失败: {exercise.get('name', 'Unknown')}, 错误: {e}"
                )

        logger.info(f"知识图谱构建完成！共处理 {total} 个练习")

    async def _create_muscles(self, muscles: List[Dict[str, Any]]):
        """创建肌群节点"""
        for muscle in muscles:
            try:
                cypher = """
                    MERGE (m:Muscle {id: $id})
                    SET m.name = $name,
                        m.name_en = $name_en,
                        m.is_front = $is_front
                """
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "id": muscle.get("id"),
                        "name": muscle.get("name", ""),
                        "name_en": muscle.get("name_en", ""),
                        "is_front": muscle.get("is_front", False),
                    },
                )
            except Exception as e:
                logger.debug(f"创建肌群失败: {e}")

    async def _create_equipment(self, equipment_list: List[Dict[str, Any]]):
        """创建器械节点"""
        for equip in equipment_list:
            try:
                cypher = """
                    MERGE (eq:Equipment {id: $id})
                    SET eq.name = $name
                """
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "id": equip.get("id"),
                        "name": equip.get("name", ""),
                    },
                )
            except Exception as e:
                logger.debug(f"创建器械失败: {e}")

    async def _create_categories(self, categories: List[Dict[str, Any]]):
        """创建类别节点"""
        for category in categories:
            try:
                cypher = """
                    MERGE (c:Category {id: $id})
                    SET c.name = $name
                """
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "id": category.get("id"),
                        "name": category.get("name", ""),
                    },
                )
            except Exception as e:
                logger.debug(f"创建类别失败: {e}")

    async def _create_exercise(self, exercise: Dict[str, Any]):
        """创建练习节点及其关系"""
        exercise_id = exercise.get("id")
        name = exercise.get("name", "")
        description = self.clean_html(exercise.get("description", ""))

        # 创建练习节点
        cypher = """
            MERGE (e:Exercise {id: $id})
            SET e.name = $name,
                e.description = $description,
                e.uuid = $uuid,
                e.created = $created,
                e.updated = $updated
        """
        self.neo4j_graph.query(
            cypher,
            params={
                "id": exercise_id,
                "name": name,
                "description": description[:2000],
                "uuid": exercise.get("uuid", ""),
                "created": exercise.get("created", ""),
                "updated": exercise.get("updated", ""),
            },
        )

        # 主肌群关系
        muscles = exercise.get("muscles", [])
        for muscle in muscles:
            cypher = """
                MATCH (e:Exercise {id: $exercise_id})
                MERGE (m:Muscle {id: $muscle_id})
                MERGE (e)-[:TARGETS_MUSCLE {is_main: true}]->(m)
            """
            try:
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "exercise_id": exercise_id,
                        "muscle_id": muscle.get("id"),
                    },
                )
            except Exception:
                pass

        # 次要肌群关系
        muscles_secondary = exercise.get("muscles_secondary", [])
        for muscle in muscles_secondary:
            cypher = """
                MATCH (e:Exercise {id: $exercise_id})
                MERGE (m:Muscle {id: $muscle_id})
                MERGE (e)-[:TARGETS_MUSCLE {is_main: false}]->(m)
            """
            try:
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "exercise_id": exercise_id,
                        "muscle_id": muscle.get("id"),
                    },
                )
            except Exception:
                pass

        # 器械关系
        equipment = exercise.get("equipment", [])
        for equip in equipment:
            cypher = """
                MATCH (e:Exercise {id: $exercise_id})
                MERGE (eq:Equipment {id: $equip_id})
                MERGE (e)-[:USES_EQUIPMENT]->(eq)
            """
            try:
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "exercise_id": exercise_id,
                        "equip_id": equip.get("id"),
                    },
                )
            except Exception:
                pass

        # 类别关系
        category = exercise.get("category")
        if category:
            cypher = """
                MATCH (e:Exercise {id: $exercise_id})
                MERGE (c:Category {id: $category_id})
                MERGE (e)-[:BELONGS_TO_CATEGORY]->(c)
            """
            try:
                self.neo4j_graph.query(
                    cypher,
                    params={
                        "exercise_id": exercise_id,
                        "category_id": category.get("id"),
                    },
                )
            except Exception:
                pass

    async def _create_constraints(self):
        """创建 Neo4j 约束"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS exercise_id ON (e:Exercise) ASSERT e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS muscle_id ON (m:Muscle) ASSERT m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS equipment_id ON (eq:Equipment) ASSERT eq.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS category_id ON (c:Category) ASSERT c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS exercise_uuid ON (e:Exercise) ASSERT e.uuid IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.neo4j_graph.query(constraint)
            except Exception as e:
                logger.debug(f"创建约束失败 (可能已存在): {e}")

        logger.info("基础约束创建完成")

    async def clear_existing_data(self):
        """清空现有数据"""
        logger.warning("清空现有数据...")

        queries = [
            "MATCH (e:Exercise)-[r]-() DELETE r",
            "MATCH (e:Exercise) DELETE e",
            "MATCH (m:Muscle) DELETE m",
            "MATCH (eq:Equipment) DELETE eq",
            "MATCH (c:Category) DELETE c",
        ]

        for query in queries:
            try:
                self.neo4j_graph.query(query)
            except Exception as e:
                logger.debug(f"清空数据时出错: {e}")

        logger.info("数据清空完成")

    def verify_graph(self) -> Dict[str, int]:
        """验证知识图谱"""
        stats = {}

        queries = [
            ("exercises", "MATCH (e:Exercise) RETURN count(e) as count"),
            ("muscles", "MATCH (m:Muscle) RETURN count(m) as count"),
            ("equipment", "MATCH (eq:Equipment) RETURN count(eq) as count"),
            ("categories", "MATCH (c:Category) RETURN count(c) as count"),
            ("relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        ]

        for name, query in queries:
            try:
                result = self.neo4j_graph.query(query)
                stats[name] = result[0]["count"] if result else 0
            except Exception as e:
                logger.error(f"查询统计失败 {name}: {e}")
                stats[name] = -1

        return stats


async def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="从 WGER 导入数据到知识图谱")
    parser.add_argument("--limit", type=int, default=None, help="限制导入的动作数量")
    parser.add_argument("--batch-size", type=int, default=50, help="批量处理大小")
    parser.add_argument("--clear", action="store_true", help="清空现有数据后导入")
    parser.add_argument("--verify", action="store_true", help="验证图谱统计信息")

    args = parser.parse_args()

    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )

    try:
        # 获取 Neo4j 连接
        neo4j_graph = get_neo4j_graph()
        logger.info(f"成功连接到 Neo4j: {settings.NEO4J_URI}")

        # 创建导入器
        ingester = WGERIngester(
            neo4j_graph=neo4j_graph,
            limit=args.limit,
            batch_size=args.batch_size,
        )

        if args.clear:
            await ingester.clear_existing_data()

        # 创建约束
        await ingester._create_constraints()

        # 获取并导入数据
        data = await ingester.fetch_all_data()
        await ingester.create_graph_nodes_and_relationships(data)

        if args.verify:
            stats = ingester.verify_graph()
            logger.info("图谱统计信息:")
            for key, value in stats.items():
                logger.info(f"  - {key}: {value}")

        logger.info("✅ WGER 数据导入完成！")

    except Exception as e:
        logger.error(f"导入失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
