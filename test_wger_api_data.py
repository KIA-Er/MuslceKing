"""
WGER API 数据测试脚本

用于测试和查看从 WGER API 爬取的数据结构：
- WGER_API_EXERCISES = https://wger.de/api/v2/exerciseinfo
- WGER_API_MUSCLES = https://wger.de/api/v2/muscle
- WGER_API_EQUIPMENT = https://wger.de/api/v2/equipment
- WGER_API_CATEGORIES = https://wger.de/api/v2/exercisecategory

使用方法:
    python test_wger_api_data.py [--output-dir DIR] [--sample-size N]
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import aiohttp
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# WGER API 基础配置
WGER_API_BASE = "https://wger.de/api/v2"
WGER_API_EXERCISES = f"{WGER_API_BASE}/exerciseinfo"
WGER_API_MUSCLES = f"{WGER_API_BASE}/muscle"
WGER_API_EQUIPMENT = f"{WGER_API_BASE}/equipment"
WGER_API_CATEGORIES = f"{WGER_API_BASE}/exercisecategory"

# WGER 静态资源基础 URL（用于图片访问）
WGER_STATIC_BASE = "https://wger.de"


class WGERAPITester:
    """WGER API 数据测试器"""

    def __init__(
        self,
        output_dir: str = "test_data_output",
        sample_size: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.data = {
            "exercises": [],
            "muscles": [],
            "equipment": [],
            "categories": [],
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
        )

    async def fetch_all_data(self) -> Dict[str, List[Dict]]:
        """并行获取所有 API 数据"""
        logger.info("开始从 WGER API 获取数据...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_data(session, WGER_API_MUSCLES, "muscles"),
                self._fetch_data(session, WGER_API_EQUIPMENT, "equipment"),
                self._fetch_data(session, WGER_API_CATEGORIES, "categories"),
                self._fetch_exercises(session),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"获取数据失败: {result}")
                else:
                    key = list(self.data.keys())[i]
                    self.data[key] = result
        
        logger.info(f"数据获取完成!")
        logger.info(f"  - 练习 (exercises): {len(self.data['exercises'])} 条")
        logger.info(f"  - 肌群 (muscles): {len(self.data['muscles'])} 条")
        logger.info(f"  - 器械 (equipment): {len(self.data['equipment'])} 条")
        logger.info(f"  - 类别 (categories): {len(self.data['categories'])} 条")
        
        return self.data

    async def _fetch_data(
        self, session: aiohttp.ClientSession, url: str, key: str
    ) -> List[Dict[str, Any]]:
        """获取单个 API 端点的数据"""
        try:
            params = {"limit": 500}  # 获取尽可能多的数据
            logger.info(f"正在获取 {key} 数据: {url}")
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"API 请求失败 {url}: {response.status}")
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                logger.info(f"  -> 获取到 {len(results)} 条 {key} 数据")
                return results
                
        except Exception as e:
            logger.error(f"获取 {key} 数据失败: {e}")
            return []

    async def _fetch_exercises(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """获取所有练习数据（带分页）"""
        all_exercises = []
        offset = 0
        batch_size = 100
        
        logger.info(f"正在获取练习数据: {WGER_API_EXERCISES}")
        
        while True:
            try:
                params = {"limit": batch_size, "offset": offset}
                async with session.get(WGER_API_EXERCISES, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"获取练习数据失败: {response.status}")
                        break
                    
                    data = await response.json()
                    exercises = data.get("results", [])
                    
                    if not exercises:
                        break
                    
                    all_exercises.extend(exercises)
                    logger.info(f"  -> 已获取 {len(all_exercises)} 个练习...")
                    
                    # 如果数据少于批次大小，说明已获取完毕
                    if len(exercises) < batch_size:
                        break
                    
                    offset += batch_size
                    
            except Exception as e:
                logger.error(f"获取练习数据失败: {e}")
                break
        
        return all_exercises

    def save_sample_data(self):
        """保存示例数据到文件"""
        logger.info(f"\n正在保存示例数据到 {self.output_dir}...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for key, data in self.data.items():
            if not data:
                continue
                
            # 保存完整数据
            full_path = self.output_dir / f"{key}_full.json"
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"  - 保存完整 {key} 数据: {full_path}")
            
            # 保存示例数据（只保存前 N 条）
            sample_path = self.output_dir / f"{key}_sample.json"
            sample_data = data[:self.sample_size]
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            logger.info(f"  - 保存 {key} 示例数据 (前 {self.sample_size} 条): {sample_path}")

    def print_data_structure(self):
        """打印数据结构分析"""
        logger.info("\n" + "="*60)
        logger.info("数据结构分析")
        logger.info("="*60)
        
        for key, data in self.data.items():
            if not data:
                continue
                
            logger.info(f"\n【{key.upper()}】")
            logger.info(f"  数据条数: {len(data)}")
            
            # 分析第一条数据的结构
            sample = data[0]
            logger.info(f"  字段列表:")
            for field, value in sample.items():
                value_type = type(value).__name__
                if isinstance(value, list):
                    if value:
                        value_type = f"List[{type(value[0]).__name__}]"
                    else:
                        value_type = "List"
                elif isinstance(value, dict):
                    value_type = "Dict"
                logger.info(f"    - {field}: {value_type}")

    def print_sample_records(self):
        """打印示例记录"""
        logger.info("\n" + "="*60)
        logger.info("示例数据记录")
        logger.info("="*60)
        
        for key, data in self.data.items():
            if not data:
                continue
                
            logger.info(f"\n【{key.upper()}】 (显示前 {min(self.sample_size, len(data))} 条):")
            logger.info("-" * 60)
            
            for i, record in enumerate(data[:self.sample_size], 1):
                logger.info(f"\n  记录 {i}:")
                # 简化显示，只显示主要字段
                main_fields = {}
                for field, value in record.items():
                    if field in ['id', 'name', 'category'] or 'name' in field.lower():
                        main_fields[field] = value
                    elif isinstance(value, str) and len(value) < 100:
                        main_fields[field] = value
                    elif isinstance(value, int) or isinstance(value, bool):
                        main_fields[field] = value
                
                for field, value in main_fields.items():
                    logger.info(f"    {field}: {value}")


async def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="测试 WGER API 数据")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data_output",
        help="输出目录 (默认: test_data_output)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="示例数据条数 (默认: 5)",
    )
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = WGERAPITester(
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )
        
        # 获取数据
        await tester.fetch_all_data()
        
        # 分析并打印数据结构
        tester.print_data_structure()
        
        # 打印示例记录
        tester.print_sample_records()
        
        # 保存数据到文件
        tester.save_sample_data()
        
        logger.info("\n" + "="*60)
        logger.info("✅ 测试完成!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())