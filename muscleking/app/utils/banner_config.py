#!/usr/bin/env python3
"""
MuscleKing Banner Config
力量感十足的启动 Banner - Rich 渐变版本
"""

import random
from rich.console import Console
from rich.text import Text

console = Console()


def interpolate_color(color1: str, color2: str, factor: float) -> str:
    """在两个十六进制颜色之间进行线性插值"""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def print_vibrant_banner(text: str):
    """
    打印带有平滑颜色渐变的横幅（灰橙渐变，体现力量感）

    Args:
        text: 要绘制的 banner 文本
    """
    lines = text.splitlines()
    if not lines:
        return

    # 灰橙渐变色板 - 从深灰到亮橙，体现力量感
    color_nodes = [
        "#1A1A1A",  # 近黑（力量根基）
        "#2D2D2D",  # 深灰
        "#404040",  # 中深灰
        "#535353",  # 中灰
        "#666666",  # 中灰
        "#7A7A7A",  # 中浅灰
        "#8D8D8D",  # 浅灰
        "#A0A0A0",  # 亮灰
        "#B8B8B8",  # 更亮灰
        "#D4A574",  # 灰橙过渡
        "#E8B87A",  # 浅橙棕
        "#FFB347",  # 橙色
        "#FFA500",  # 标准橙
        "#FF8C00",  # 深橙
        "#FF6600",  # 橙红
        "#FF4500",  # 红橙（能量爆发）
    ]

    num_nodes = len(color_nodes)

    for line in lines:
        rich_text = Text()
        line_len = len(line)

        if line_len <= 1:
            console.print(line)
            continue

        for i, char in enumerate(line):
            # 1. 计算全局进度 (0.0 到 1.0)
            relative_pos = i / (line_len - 1) if line_len > 1 else 0

            # 2. 确定当前进度落在哪个颜色区间内
            scaled_pos = relative_pos * (num_nodes - 1)
            index = int(scaled_pos)
            next_index = min(index + 1, num_nodes - 1)

            # 3. 计算在小区间内的局部进度 (0.0 到 1.0)
            local_factor = scaled_pos - index

            # 4. 获取插值颜色
            if index == next_index:
                color = color_nodes[-index]
            else:
                color = interpolate_color(
                    color_nodes[-index], color_nodes[-next_index], local_factor
                )

            rich_text.append(char, style=f"bold {color}")

        console.print(rich_text)


def print_loading_message(message: str = "初始化系统...") -> None:
    """打印加载消息"""
    console.print(f"⚡ [bold gold]{message}[/]")


def print_power_quote() -> None:
    """打印随机励志名言"""
    quotes = [
        "No pain, no gain - 没有付出，就没有收获",
        "强者不是赢过别人，而是赢过昨天的自己",
        "Strength does not come from physical capacity. It comes from an indomitable will.",
        "The only bad workout is the one that didn't happen",
        "Discipline is doing what needs to be done, even if you don't want to do it",
        "Your body can stand almost anything. It's your mind that you have to convince",
        "今天不想练，明天更不想练，后天就成了别人的垫脚石",
    ]
    quote = random.choice(quotes)
    console.print(f"\n[bold orange1]  💬 {quote}[/]\n")


def start_banner(text: str, message: str = "系统启动中"):
    print_vibrant_banner(text)
    print_loading_message(message)
    print_power_quote()
