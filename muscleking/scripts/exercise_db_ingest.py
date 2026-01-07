"""
exerciseDB 数据导入脚本

从 exerciseDB API 获取健身动作数据，提取节点和关系，生成知识图谱。
数据来源: https://github.com/yuhonas/exerciseDB

使用方法:
    python scripts/exercise_db_ingest.py [--limit N] [--batch-size N] [--clear]
"""
import asyncio
import argparse
import json
import aiohttp
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from loguru import logger
import sys

# 添加项目根目录到路径
# 脚本在 muscleking/scripts/ 下，需要向上2级到达项目根目录
scripts_dir = Path(__file__).resolve().parent
scripts_parent = scripts_dir.parent  # muscleking/
project_root = scripts_parent.parent  # 项目根目录

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_parent))

# 尝试从 settings 获取配置，如果不可用则使用默认值
try:
    from muscleking.config import settings
    NEO4J_URI = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = getattr(settings, 'NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = getattr(settings, 'NEO4J_PASSWORD', 'muscleking')
    NEO4J_DATABASE = getattr(settings, 'NEO4J_DATABASE', 'neo4j')
except ImportError:
    NEO4J_URI = 'bolt://localhost:7687'
    NEO4J_USER = 'neo4j'
    NEO4J_PASSWORD = 'muscleking'
    NEO4J_DATABASE = 'neo4j'

from muscleking.app.persistence.core.neo4jconn import get_neo4j_graph


# exerciseDB API 基础 URL
EXERCISE_DB_BASE_URL = "https://raw.githubusercontent.com/yuhonas/exerciseDB/main/exercises"


class ExerciseDBIngester:
    """exerciseDB 数据导入器"""

    def __init__(
        self,
        neo4j_graph,
        limit: Optional[int] = None,
        batch_size: int = 50,
    ):
        self.neo4j_graph = neo4j_graph
        self.limit = limit
        self.batch_size = batch_size
        
        # 已存在的节点集合，用于避免重复创建
        self.existing_muscles: Set[str] = set()
        self.existing_equipment: Set[str] = set()
        self.existing_difficulties: Set[str] = {"easy", "medium", "hard"}
        self.existing_goals: Set[str] = {"增肌", "减脂", "力量", "塑形", "心肺"}
        self.existing_risks: Set[str] = set()
        self.existing_benefits: Set[str] = set()

    async def fetch_exercises(self) -> List[Dict[str, Any]]:
        """从 exerciseDB 获取所有动作数据"""
        logger.info("正在从 exerciseDB 获取数据...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{EXERCISE_DB_BASE_URL}.json") as response:
                    if response.status != 200:
                        raise Exception(f"API 请求失败: {response.status}")
                    data = await response.json()
                    
                    exercises = data.get("exercises", []) or data
                    if self.limit:
                        exercises = exercises[:self.limit]
                    
                    logger.info(f"获取到 {len(exercises)} 个训练动作")
                    return exercises
            except Exception as e:
                logger.error(f"获取 exerciseDB 数据失败: {e}")
                raise

    def parse_exercise(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析单个动作数据为知识图谱格式"""
        # 标准化字段名
        name = raw_data.get("name", "")
        
        # 解析设备
        equipment_list = self._normalize_list(raw_data.get("equipment", []))
        if isinstance(equipment_list, str):
            equipment_list = [equipment_list]
        
        # 解析身体部位
        body_parts = self._normalize_list(raw_data.get("bodyPart", []))
        if isinstance(body_parts, str):
            body_parts = [body_parts]
        
        # 解析目标肌肉
        target_muscles = self._normalize_list(raw_data.get("targetMuscle", []))
        if isinstance(target_muscles, str):
            target_muscles = [target_muscles]
        
        # 解析辅助肌肉
        secondary_muscles = self._normalize_list(raw_data.get("secondaryMuscles", []))
        if isinstance(secondary_muscles, str):
            secondary_muscles = [secondary_muscles]
        
        # 解析指导步骤
        instructions = self._normalize_list(raw_data.get("instructions", []))
        if isinstance(instructions, str):
            instructions = [instructions]
        
        return {
            "id": raw_data.get("id", name),
            "name": name,
            "description": raw_data.get("instructions", [])[0] if instructions else "",
            "equipment": equipment_list,
            "body_parts": body_parts,
            "target_muscles": list(set(target_muscles + secondary_muscles)),
            "secondary_muscles": secondary_muscles,
            "instructions": instructions,
            "gif_url": raw_data.get("gifUrl", ""),
        }

    def _normalize_list(self, value: Any) -> List[str]:
        """标准化各种输入为字符串列表"""
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        elif isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return []

    def _to_camel_case(self, text: str) -> str:
        """转换为驼峰命名"""
        words = text.lower().split()
        if not words:
            return text
        return words[0] + "".join(w.capitalize() for w in words[1:])

    async def create_graph_nodes_and_relationships(self, exercise: Dict[str, Any]) -> List[str]:
        """为单个动作创建节点和关系的 Cypher 语句"""
        cypher_statements = []
        
        name = exercise["name"]
        exercise_id = exercise["id"]
        
        # 1. 创建 Exercise 节点
        cypher_statements.append(f"""
            MERGE (e:Exercise {{id: $exercise_id}})
            SET e.name = $name,
                e.description = $description,
                e.calories_per_min = $calories
        """)
        
        # 2. 创建 Equipment 节点和关系
        for equip in exercise["equipment"]:
            equip_normalized = self._normalize_equipment(equip)
            self.existing_equipment.add(equip_normalized)
            cypher_statements.append(f"""
                MERGE (eq:Equipment {{name: $equip_name}})
                WITH e, eq
                MERGE (e)-[:USES_EQUIPMENT]->(eq)
            """)
        
        # 3. 创建 Muscle 节点和关系 (目标肌群)
        for muscle in exercise["target_muscles"]:
            muscle_normalized = self._normalize_muscle(muscle)
            self.existing_muscles.add(muscle_normalized)
            cypher_statements.append(f"""
                MERGE (m:Muscle {{name: $muscle_name}})
                WITH e, m
                MERGE (e)-[:TARGETS_MUSCLE]->(m)
            """)
        
        # 4. 创建 Difficulty 节点和关系
        difficulty = self._estimate_difficulty(exercise)
        cypher_statements.append(f"""
            MERGE (d:Difficulty {{name: $difficulty}})
            WITH e, d
            MERGE (e)-[:HAS_DIFFICULTY]->(d)
        """)
        
        # 5. 创建 ExerciseStep 节点和关系
        for idx, instruction in enumerate(exercise["instructions"], 1):
            cypher_statements.append(f"""
                MERGE (s:ExerciseStep {{id: $step_id}})
                SET s.order = $order,
                    s.instruction = $instruction
                WITH e, s
                MERGE (e)-[:HAS_STEP]->(s)
            """)
        
        # 6. 创建 TrainingGoal 节点和关系
        for goal in self._estimate_goals(exercise):
            self.existing_goals.add(goal)
            cypher_statements.append(f"""
                MERGE (g:TrainingGoal {{name: $goal}})
                WITH e, g
                MERGE (e)-[:SUPPORTS_GOAL]->(g)
            """)
        
        return cypher_statements

    def _normalize_equipment(self, equip: str) -> str:
        """标准化器械名称"""
        equip_lower = equip.lower().strip()
        
        equipment_mapping = {
            "body": "自重",
            "body weight": "自重",
            "dumbbell": "哑铃",
            "barbell": "杠铃",
            "kettlebell": "壶铃",
            "cable": "滑轮机",
            "machine": "器械",
            "ez bar": "EZ杠",
            "resistance band": "弹力带",
            "medicine ball": "药球",
        }
        
        return equipment_mapping.get(equip_lower, equip)

    def _normalize_muscle(self, muscle: str) -> str:
        """标准化肌群名称"""
        muscle_lower = muscle.lower().strip()
        
        muscle_mapping = {
            "chest": "胸肌",
            "pectorals": "胸肌",
            "upper chest": "上胸肌",
            "back": "背部",
            "latissimus dorsi": "背阔肌",
            "traps": "斜方肌",
            "trapezius": "斜方肌",
            "shoulders": "肩部",
            "deltoids": "三角肌",
            "anterior deltoid": "前三角肌",
            "posterior deltoid": "后三角肌",
            "lateral deltoid": "侧三角肌",
            "biceps": "肱二头肌",
            "triceps": "肱三头肌",
            "forearms": "前臂",
            "quadriceps": "股四头肌",
            "quads": "股四头肌",
            "hamstrings": "腘绳肌",
            "glutes": "臀部",
            "gluteus": "臀部",
            "calves": "小腿",
            "gastrocnemius": "腓肠肌",
            "soleus": "比目鱼肌",
            "abs": "腹部",
            "abdominals": "腹肌",
            "core": "核心",
            "lower back": "下背部",
            "lower back": "腰部",
            "erector spinae": "竖脊肌",
            "obliques": "腹斜肌",
            "serratus": "锯齿肌",
            "hip flexors": "髋部屈肌",
            "adductors": "内收肌",
        }
        
        return muscle_mapping.get(muscle_lower, muscle)

    def _estimate_difficulty(self, exercise: Dict[str, Any]) -> str:
        """根据动作特征估算难度"""
        name = exercise["name"].lower()
        
        # 简单规则估算
        difficult_keywords = ["爆发", "爆发力", "奥林匹克", "挺举", "抓举", "硬拉", "深蹲", "卧推"]
        easy_keywords = ["拉伸", "热身", "放松", "收缩", "激活"]
        
        if any(kw in name for kw in difficult_keywords):
            return "hard"
        elif any(kw in name for kw in easy_keywords):
            return "easy"
        else:
            return "medium"

    def _estimate_goals(self, exercise: Dict[str, Any]) -> List[str]:
        """根据动作类型估算训练目标"""
        name = exercise["name"].lower()
        equipment = exercise.get("equipment", [])
        
        goals = ["增肌", "力量"]
        
        # 有氧相关
        if any(kw in name for kw in ["跑步", "跳绳", "登山", "跳跃", "波比"]):
            goals.append("减脂")
            goals.append("心肺")
        
        # 核心训练
        if any(kw in name for kw in ["平板", "卷腹", "桥", "死虫", "俄罗斯转体"]):
            goals.append("核心")
        
        return goals

    async def import_to_neo4j(self, exercises: List[Dict[str, Any]]):
        """将解析后的数据导入 Neo4j"""
        logger.info("开始导入数据到 Neo4j...")
        
        total = len(exercises)
        
        # 先创建基础约束
        await self._create_constraints()
        
        for i, raw_exercise in enumerate(exercises):
            try:
                exercise = self.parse_exercise(raw_exercise)
                cypher_statements = await self.create_graph_nodes_and_relationships(exercise)
                
                # 批量执行
                for stmt in cypher_statements:
                    try:
                        self.neo4j_graph.query(stmt, params={
                            "exercise_id": exercise["id"],
                            "name": exercise["name"],
                            "description": exercise["description"][:500] if exercise["description"] else "",
                            "calories": 5,  # 默认值
                        })
                    except Exception as e:
                        logger.debug(f"执行 Cypher 语句失败: {e}")
                
                if (i + 1) % self.batch_size == 0:
                    logger.info(f"进度: {i + 1}/{total} ({100*(i+1)//total}%)")
                    
            except Exception as e:
                logger.warning(f"导入动作失败: {exercise.get('name', 'Unknown')}, 错误: {e}")
        
        logger.info(f"导入完成！共处理 {total} 个动作")

    async def _create_constraints(self):
        """创建 Neo4j 约束"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS exercise_id ON (e:Exercise) ASSERT e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS exercise_name ON (e:Exercise) ASSERT e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS muscle_name ON (m:Muscle) ASSERT m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS equipment_name ON (eq:Equipment) ASSERT eq.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS difficulty_name ON (d:Difficulty) ASSERT d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS goal_name ON (g:TrainingGoal) ASSERT g.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS step_id ON (s:ExerciseStep) ASSERT s.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                self.neo4j_graph.query(constraint)
            except Exception as e:
                logger.debug(f"创建约束失败 (可能已存在): {e}")
        
        logger.info("基础约束创建完成")


async def main():
    """主入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从 exerciseDB 导入数据到知识图谱")
    parser.add_argument("--limit", type=int, default=None, help="限制导入的动作数量")
    parser.add_argument("--batch-size", type=int, default=50, help="批量处理大小")
    parser.add_argument("--clear", action="store_true", help="清空现有数据后导入")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    try:
        # 获取 Neo4j 连接
        neo4j_graph = get_neo4j_graph()
        logger.info(f"成功连接到 Neo4j: {settings.NEO4J_URI}")
        
        # 创建导入器
        ingester = ExerciseDBIngester(
            neo4j_graph=neo4j_graph,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        
        if args.clear:
            logger.warning("清空模式: 将删除所有 Exercise 相关的节点和关系")
            neo4j_graph.query("MATCH (e:Exercise)-[r]-() DELETE r")
            neo4j_graph.query("MATCH (e:Exercise) DELETE e")
            neo4j_graph.query("MATCH (m:Muscle) WHERE NOT (m)--() DELETE m")
            neo4j_graph.query("MATCH (eq:Equipment) WHERE NOT (eq)--() DELETE eq")
        
        # 获取并导入数据
        exercises = await ingester.fetch_exercises()
        await ingester.import_to_neo4j(exercises)
        
        logger.info("✅ exerciseDB 数据导入完成！")
        
    except Exception as e:
        logger.error(f"导入失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())