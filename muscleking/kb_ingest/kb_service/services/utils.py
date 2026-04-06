from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Dict, Mapping

import pandas as pd

from kb_service.core.config import Config


def flatten_row(row: Mapping[str, object], config: Config) -> str:
    """处理Excel的第二种做法:扁平化
    将一行字典转换为扁平化的 key: value 字符串，用于生成嵌入向量.

    Args:
        row: 要处理的字典，键为列名，值为单元格内容.
        config: 配置对象，包含扁平化相关的参数.

    Returns:
        扁平化的键值对字符串，格式为 "key1:value1 | key2:value2 | ..."

    Example:
    输入：
        row = {
            "菜名": "红烧肉",
            "口味 (咸度)": "咸鲜",
            "难度": "中等",
            "备注": "",
            "Unnamed: 5": None,
            "烹饪时间": pd.Timestamp("2024-01-01 12:00:00")
        }
    输出：
        "菜名:红烧肉 | 口味:咸鲜 | 难度:中等 | 烹饪时间:2024-01-01 12:00:00"
    """
    # 存储最终的扁平化键值对字符串
    flat_pairs: list[str] = []
    # 记录已处理的键值对，避免重复
    seen: Dict[str, object] = {}
    # 第一遍遍历: 键名标准化和去重
    for key, value in row.items():
        # 键名预处理
        if key is None:
            continue
        normalized_key = str(key).strip().replace("（", "(").replace("）", ")")
        while True:
            stripped = re.sub(r"\s*\([^()]*\)\s*$", "", normalized_key)
            if stripped == normalized_key:
                break
            normalized_key = stripped
        # 跳过未命名列
        if re.match(r"^Unnamed[:\s]", normalized_key, flags=re.I):
            continue
        # 处理重复键(条件：键未出现过 或 旧的值是空的而新值非空)
        if normalized_key not in seen or (
            not _nonempty(seen[normalized_key]) and _nonempty(value)
        ):
            if isinstance(value, (pd.Timestamp, datetime, date)):
                value = pd.Timestamp(value).isoformat(sep=" ", timespec="seconds")
            seen[normalized_key] = value
    # 第二遍遍历: 生成键值对字符串
    for normalized_key, value in seen.items():
        if value is None:
            continue
        try:
            if not pd.notna(value):
                continue
        except Exception:  # pragma: no cover
            pass
        value_str = str(value).strip()
        if value_str in {"", "-", "nan", "NaN"}:
            continue
        # 生成键值对字符串
        # 例子: "菜名:红烧肉",其中菜名是key，红烧肉是value
        flat_pairs.append(f"{normalized_key}{config.flat_kv_sep}{value_str}")
    # 用分隔符连接所有字符串
    # 例子: 分隔符为" | "", 则 flat_text = "菜名:红烧肉 | 口味:咸鲜 | 难度:中等"
    flat_text = config.flat_sep.join(flat_pairs)
    if config.flat_max_len > 0 and len(flat_text) > config.flat_max_len:
        flat_text = flat_text[: config.flat_max_len]
    return flat_text


def _nonempty(value) -> bool:
    """判断一个值是否为"有效值"(非None、非空字符串、非NaN等)"""
    if value is None:
        return False
    try:
        if not pd.notna(value):
            return False
    except Exception:  # pragma: no cover
        pass
    return str(value).strip() not in {"", "-", "nan", "NaN", "None", "null"}


def compute_content_hash(data: Dict | str, algorithm: str = "md5") -> str:
    """
    计算数据的哈希值，用于检测内容变化

    Args:
        data: 字典或字符串数据
        algorithm: 哈希算法，支持 'md5', 'sha256'

    Returns:
        哈希值的十六进制字符串
    """
    if isinstance(data, dict):
        # 将字典转为稳定的JSON字符串（排序key）
        content = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    else:
        content = str(data)

    # 创建哈希对象
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()
    # 将内容编码为字节(因为哈希算法只能处理字节，不能直接处理字符串)并更新哈希计算
    hasher.update(content.encode("utf-8"))
    # 完成计算并返回哈希计算结果的十六进制字符
    return hasher.hexdigest()
