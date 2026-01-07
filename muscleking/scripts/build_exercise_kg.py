"""
知识图谱构建脚本

从本地 JSON 文件构建健身动作知识图谱，支持增量更新。
用于将 exerciseDB 数据转换为 Neo4j 知识图谱。

使用方法:
    python scripts/build_exercise_kg.py --input exercises.json [--clear]
    python scripts/build_exercise_kg.py --demo  # 使用内置示例数据
"""
import asyncio
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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


class ExerciseKGBuilder:
    """健身动作知识图谱构建器"""

    def __init__(
        self,
        neo4j_graph,
        batch_size: int = 50,
    ):
        self.neo4j_graph = neo4j_graph
        self.batch_size = batch_size

    def load_exercises(self, file_path: str) -> List[Dict[str, Any]]:
        """从 JSON 文件加载动作数据"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        exercises = data.get("exercises", []) or data
        logger.info(f"从 {file_path} 加载了 {len(exercises)} 个动作")
        return exercises

    def get_demo_exercises(self) -> List[Dict[str, Any]]:
        """获取内置示例数据"""
        return [
            {
                "id": "001",
                "name": "杠铃深蹲",
                "equipment": ["barbell"],
                "bodyPart": ["legs"],
                "targetMuscle": ["quadriceps"],
                "secondaryMuscles": ["glutes", "hamstrings", "core"],
                "instructions": [
                    "双脚与肩同宽站立，脚尖略微外展",
                    "将杠铃放在上斜方肌上，保持背部挺直",
                    "吸气的同时屈膝下蹲，膝盖跟随脚尖方向",
                    "下蹲至大腿与地面平行或略低",
                    "呼气的同时通过脚跟发力站起",
                    "重复指定次数"
                ],
                "gifUrl": ""
            },
            {
                "id": "002",
                "name": "平板卧推",
                "equipment": ["bench", "barbell"],
                "bodyPart": ["chest"],
                "targetMuscle": ["pectorals"],
                "secondaryMuscles": ["triceps", "anterior deltoid"],
                "instructions": [
                    "仰卧于水平长椅上，双脚平放地面",
                    "握住杠铃，手距略宽于肩宽",
                    "将杠铃缓慢下放至胸部中下部位置",
                    "推起杠铃至手臂完全伸直",
                    "控制离心阶段，下放时不要借助反弹",
                    "重复指定次数"
                ],
                "gifUrl": ""
            },
            {
                "id": "003",
                "name": "硬拉",
                "equipment": ["barbell"],
                "bodyPart": ["back", "legs"],
                "targetMuscle": ["hamstrings", "glutes", "lower back"],
                "secondaryMuscles": ["core", "forearms"],
                "instructions": [
                    "双脚与髋同宽站立，杠铃贴近小腿",
                    "屈髋屈膝握住杠铃，保持背部平直",
                    "深吸一口气收紧核心",
                    "通过伸髋伸膝将杠铃提起",
                    "站直后髋部前送，肩部后展",
                    "缓慢放下杠铃回到起始位置",
                    "重复指定次数"
                ],
                "gifUrl": ""
            },
            {
                "id": "004",
                "name": "引体向上",
                "equipment": ["pull-up bar"],
                "bodyPart": ["back"],
                "targetMuscle": ["latissimus dorsi"],
                "secondaryMuscles": ["biceps", "core"],
                "instructions": [
                    "双手正握杠杆，距离略宽于肩宽",
                    "身体自然悬挂，手臂完全伸直",
                    "吸气的同时屈肘将身体向上拉",
                    "下巴超过杠杆顶部",
                    "呼气的同时缓慢放下身体",
                    "不要完全放松，保持肌肉张力"
                ],
                "gifUrl": ""
            },
            {
                "id": "005",
                "name": "哑铃肩上推举",
                "equipment": ["dumbbell"],
                "bodyPart": ["shoulders"],
                "targetMuscle": ["deltoids"],
                "secondaryMuscles": ["triceps"],
                "instructions": [
                    "双脚与肩同宽站立，双手各握一只哑铃",
                    "将哑铃举至肩部上方，手掌朝前",
                    "保持背部挺直，核心收紧",
                    "将哑铃向上推举直至手臂完全伸直",
                    "缓慢下放至起始位置",
                    "重复指定次数"
                ],
                "gifUrl": ""
            },
            {
                "id": "006",
                "name": "波比跳",
                "equipment": [],
                "bodyPart": ["full body"],
                "targetMuscle": ["quadriceps", "chest", "core"],
                "secondaryMuscles": ["hamstrings", "shoulders"],
                "instructions": [
                    "站立姿势，双脚与肩同宽",
                    "下蹲至双手触地",
                    "双脚向后跳至俯卧撑姿势",
                    "完成一个俯卧撑",
                    "双脚向前跳回下蹲姿势",
                    "爆发式跳起并拍手",
                    "落地后立即进入下一个循环"
                ],
                "gifUrl": ""
            },
            {
                "id": "007",
                "name": "坐姿划船",
                "equipment": ["cable", "machine"],
                "bodyPart": ["back"],
                "targetMuscle": ["upper back", "rhomboids"],
                "secondaryMuscles": ["biceps", "lats"],
                "instructions": [
                    "坐在划船机座位上，双脚踩在踏板上",
                    "握住把手，保持背部挺直",
                    "呼气的同时将把手拉向腹部",
                    "挤压肩胛骨，感受背部发力",
                    "吸气的同时缓慢伸直手臂",
                    "保持控制，不要借助惯性"
                ],
                "gifUrl": ""
            },
            {
                "id": "008",
                "name": "腘绳肌弯举",
                "equipment": ["leg curl machine"],
                "bodyPart": ["legs"],
                "targetMuscle": ["hamstrings"],
                "secondaryMuscles": [],
                "instructions": [
                    "俯卧在腿弯举机器上",
                    "调整滚轴位置使其正好位于脚踝上方",
                    "吸气的同时弯曲膝盖向上抬腿",
                    "感受腘绳肌的收缩",
                    "在最高点保持1-2秒",
                    "呼气的同时缓慢放下双腿",
                    "不要完全伸直膝盖"
                ],
                "gifUrl": ""
            },
            {
                "id": "009",
                "name": "平板支撑",
                "equipment": [],
                "bodyPart": ["core"],
                "targetMuscle": ["core", "abs"],
                "secondaryMuscles": ["shoulders"],
                "instructions": [
                    "俯卧姿势，前臂和肘部支撑身体",
                    "身体呈一条直线，从头部到脚跟",
                    "收紧腹部和臀部肌肉",
                    "保持正常呼吸，不要憋气",
                    "目光向下看地面",
                    "保持指定时间"
                ],
                "gifUrl": ""
            },
            {
                "id": "010",
                "name": "跳绳",
                "equipment": ["jump rope"],
                "bodyPart": ["full body"],
                "targetMuscle": ["calves", "quadriceps"],
                "secondaryMuscles": ["core", "shoulders"],
                "instructions": [
                    "双手握住跳绳手柄，绳子置于身后",
                    "双脚并拢站立，膝盖微曲",
                    "手腕画圈甩动跳绳",
                    "双脚同时跳过绳子",
                    "保持节奏稳定",
                    "落地时膝盖微曲以减少冲击"
                ],
                "gifUrl": ""
            }
        ]

    def _normalize_equipment(self, equip: str) -> str:
        """标准化器械名称"""
        equip_lower = str(equip).lower().strip()
        
        equipment_mapping = {
            "body": "自重",
            "body weight": "自重",
            "dumbbell": "哑铃",
            "barbell": "杠铃",
            "kettlebell": "壶铃",
            "cable": "滑轮机",
            "machine": "器械",
            "machine": "器械",
            "ez bar": "EZ杠",
            "resistance band": "弹力带",
            "medicine ball": "药球",
            "bench": "长椅",
            "pull-up bar": "引体向上杆",
            "jump rope": "跳绳",
            "leg curl machine": "腿弯举机",
        }
        
        return equipment_mapping.get(equip_lower, equip)

    def _normalize_muscle(self, muscle: str) -> str:
        """标准化肌群名称"""
        muscle_lower = str(muscle).lower().strip()
        
        muscle_mapping = {
            "chest": "胸肌",
            "pectorals": "胸肌",
            "upper chest": "上胸肌",
            "upper back": "上背部",
            "back": "背部",
            "latissimus dorsi": "背阔肌",
            "lats": "背阔肌",
            "traps": "斜方肌",
            "trapezius": "斜方肌",
            "rhomboids": "菱形肌",
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
            "legs": "腿部",
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
            "full body": "全身",
        }
        
        return muscle_mapping.get(muscle_lower, muscle)

    def _estimate_difficulty(self, exercise: Dict[str, Any], name: str) -> str:
        """根据动作特征估算难度"""
        name_lower = name.lower()
        
        # 复杂/高风险动作
        difficult_keywords = ["硬拉", "深蹲", "卧推", "划船", "推举", "引体向上", "波比跳", "奥林匹克", "挺举", "抓举"]
        # 简单动作
        easy_keywords = ["拉伸", "热身", "放松", "收缩", "弯举", "卷腹", "支撑"]
        
        if any(kw in name for kw in difficult_keywords):
            return "hard"
        elif any(kw in name for kw in easy_keywords):
            return "easy"
        else:
            return "medium"

    def _estimate_goals(self, name: str) -> List[str]:
        """根据动作名称估算训练目标"""
        name_lower = name.lower()
        goals = ["增肌", "力量"]
        
        # 有氧/减脂
        if any(kw in name for kw in ["跳绳", "登山", "跳跃", "波比", "跑"]):
            goals.append("减脂")
            goals.append("心肺")
        
        # 核心
        if any(kw in name for kw in ["平板", "卷腹", "支撑", "桥"]):
            goals.append("核心")
        
        return goals

    async def build_kg(self, exercises: List[Dict[str, Any]]):
        """构建知识图谱"""
        logger.info("开始构建知识图谱...")
        
        # 创建约束
        await self._create_constraints()
        
        total = len(exercises)
        created_count = 0
        
        for i, raw_exercise in enumerate(exercises):
            try:
                # 解析数据
                name = raw_exercise.get("name", f"Exercise_{i}")
                exercise_id = raw_exercise.get("id", f"exercise_{i}")
                
                # 获取设备列表
                equipment_list = raw_exercise.get("equipment", [])
                if isinstance(equipment_list, str):
                    equipment_list = [equipment_list]
                equipment_list = [self._normalize_equipment(e) for e in equipment_list]
                
                # 获取肌群列表
                target_muscles = raw_exercise.get("targetMuscle", [])
                if isinstance(target_muscles, str):
                    target_muscles = [target_muscles]
                secondary_muscles = raw_exercise.get("secondaryMuscles", [])
                if isinstance(secondary_muscles, str):
                    secondary_muscles = [secondary_muscles]
                
                all_muscles = list(set(target_muscles + secondary_muscles))
                muscles_list = [self._normalize_muscle(m) for m in all_muscles]
                
                # 获取步骤
                instructions = raw_exercise.get("instructions", [])
                if isinstance(instructions, str):
                    instructions = [instructions]
                
                # 估算难度和目标
                difficulty = self._estimate_difficulty(raw_exercise, name)
                goals = self._estimate_goals(name)
                
                # 构建 Cypher
                cypher = self._build_cypher(
                    exercise_id=exercise_id,
                    name=name,
                    description=instructions[0] if instructions else "",
                    equipment=equipment_list,
                    muscles=muscles_list,
                    instructions=instructions,
                    difficulty=difficulty,
                    goals=goals,
                )
                
                # 执行
                self.neo4j_graph.query(cypher)
                created_count += 1
                
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    logger.info(f"进度: {i + 1}/{total} ({100*(i+1)//total}%) - 已创建 {created_count} 个动作")
                    
            except Exception as e:
                logger.warning(f"处理动作失败: {name}, 错误: {e}")
        
        logger.info(f"✅ 知识图谱构建完成！共创建 {created_count} 个动作节点")

    def _build_cypher(
        self,
        exercise_id: str,
        name: str,
        description: str,
        equipment: List[str],
        muscles: List[str],
        instructions: List[str],
        difficulty: str,
        goals: List[str],
    ) -> str:
        """构建单个动作的 Cypher 语句"""
        
        # 先转义字符串中的单引号
        safe_name = name.replace("'", "''")
        safe_desc = description.replace("'", "''")[:500] if description else ""
        
        # 构建 Exercise 节点
        cypher = f"""
        MERGE (e:Exercise {{id: '{exercise_id}'}})
        SET e.name = '{safe_name}',
            e.description = '{safe_desc}',
            e.calories_per_min = 5
        """
        
        # 设备关系
        for equip in equipment:
            if equip:
                safe_equip = equip.replace("'", "''")
                cypher += f"""
                MERGE (eq:Equipment {{name: '{safe_equip}'}})
                MERGE (e)-[:USES_EQUIPMENT]->(eq)
                """
        
        # 肌群关系
        for muscle in muscles:
            if muscle:
                safe_muscle = muscle.replace("'", "''")
                cypher += f"""
                MERGE (m:Muscle {{name: '{safe_muscle}'}})
                MERGE (e)-[:TARGETS_MUSCLE]->(m)
                """
        
        # 难度关系
        cypher += f"""
        MERGE (d:Difficulty {{name: '{difficulty}'}})
        MERGE (e)-[:HAS_DIFFICULTY]->(d)
        """
        
        # 训练目标关系
        for goal in goals:
            if goal:
                safe_goal = goal.replace("'", "''")
                cypher += f"""
                MERGE (g:TrainingGoal {{name: '{safe_goal}'}})
                MERGE (e)-[:SUPPORTS_GOAL]->(g)
                """
        
        # 步骤关系
        for idx, instruction in enumerate(instructions, 1):
            if instruction:
                step_id = f"{exercise_id}_step_{idx}"
                safe_instruction = instruction.replace("'", "''")[:200]
                cypher += f"""
                MERGE (s:ExerciseStep {{id: '{step_id}'}})
                SET s.order = {idx},
                    s.instruction = '{safe_instruction}'
                MERGE (e)-[:HAS_STEP]->(s)
                """
        
        return cypher

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

    async def clear_existing_data(self):
        """清空现有数据"""
        logger.warning("清空现有数据...")
        
        queries = [
            "MATCH (e:Exercise)-[r]-() DELETE r",
            "MATCH (e:Exercise) DELETE e",
            "MATCH (m:Muscle) WHERE NOT (m)--() DELETE m",
            "MATCH (eq:Equipment) WHERE NOT (eq)--() DELETE eq",
            "MATCH (d:Difficulty) WHERE NOT (d)--() DELETE d",
            "MATCH (g:TrainingGoal) WHERE NOT (g)--() DELETE g",
            "MATCH (s:ExerciseStep) WHERE NOT (s)--() DELETE s",
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
    parser = argparse.ArgumentParser(description="构建健身动作知识图谱")
    parser.add_argument("--input", type=str, help="输入 JSON 文件路径")
    parser.add_argument("--demo", action="store_true", help="使用内置示例数据")
    parser.add_argument("--clear", action="store_true", help="清空现有数据后导入")
    parser.add_argument("--verify", action="store_true", help="验证图谱统计信息")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    if not args.input and not args.demo:
        parser.error("必须指定 --input 或 --demo 参数")
        return
    
    try:
        # 获取 Neo4j 连接
        neo4j_graph = get_neo4j_graph()
        logger.info(f"成功连接到 Neo4j: {settings.NEO4J_URI}")
        
        # 创建构建器
        builder = ExerciseKGBuilder(neo4j_graph=neo4j_graph)
        
        if args.clear:
            await builder.clear_existing_data()
        
        # 加载数据
        if args.demo:
            exercises = builder.get_demo_exercises()
            logger.info("使用内置示例数据")
        else:
            exercises = builder.load_exercises(args.input)
        
        # 构建图谱
        await builder.build_kg(exercises)
        
        # 验证
        if args.verify or args.demo:
            stats = builder.verify_graph()
            logger.info("图谱统计信息:")
            for key, value in stats.items():
                logger.info(f"  - {key}: {value}")
        
        logger.info("✅ 知识图谱构建完成！")
        
    except Exception as e:
        logger.error(f"构建失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())