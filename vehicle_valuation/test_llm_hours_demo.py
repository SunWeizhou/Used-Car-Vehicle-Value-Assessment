#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 工时辅助判定功能 - 演示版

展示已有的 LLM 结果，验证工时辅助判定的效果
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data


def main():
    """演示 LLM 工时辅助判定效果"""
    print("\n" + "="*100)
    print("LLM 工时辅助判定 - 功能演示")
    print("="*100 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载原始数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 检查是否有 LLM 结果文件
    llm_result_path = Path(__file__).parent / "data" / "llm_parsed_results.csv"

    if not llm_result_path.exists():
        print("\n⚠️  未找到 LLM 结果文件")
        print(f"   期望路径: {llm_result_path}")
        print("\n如需生成 LLM 结果，请运行:")
        print("  export DEEPSEEK_API_KEY='your-api-key'")
        print("  python -c \"from utils.llm_structuring import process_sample_batch; ...\"")
        return

    # 3. 加载 LLM 结果
    print("\n步骤 2: 加载 LLM 结果")
    df_llm = pd.read_csv(llm_result_path)
    print(f"  ✓ 加载 {len(df_llm)} 条 LLM 分析结果\n")

    # 4. 展示工时辅助判定规则
    print("="*100)
    print("【工时辅助判定规则】")
    print("="*100)
    print("""
请结合工时 (Repair Hours) 进行动态修正：
1. [排除误报]: 即使涉及核心部件关键词(如发动机)，若工时极短(<1h)，
   通常为检查/调整，应降级为 L1。
2. [确认重症]: 涉及核心部件且工时显著(>8h)，
   通常为解体维修/总成更换，应确认为 L3。
3. [区分工种]: 高工时的"喷漆/钣金"属于 L1/L2 (车身)，
   高工时的"机械拆装"属于 L3 (事故/大修)。
    """)

    # 5. 展示前 10 条结果的详细信息
    print("="*100)
    print("【LLM 分析结果展示 - 前 10 条】")
    print("="*100 + "\n")

    sample_results = df_llm.head(10)

    for idx, row in sample_results.iterrows():
        record_id = row['ID']

        # 获取对应的原始数据
        base_record = df_base[df_base['ID'] == record_id]
        time_records = df_time[df_time['RECORD_ID'] == record_id]
        parts_records = df_parts[df_parts['RECORD_ID'] == record_id]

        print(f"\n{'─'*100}")
        print(f"记录 ID: {record_id}")
        print(f"{'─'*100}")

        # 显示原始维修信息
        if not base_record.empty:
            fault_desc = base_record.iloc[0].get('FAULT_DESC', '无')
            print(f"📋 故障描述: {fault_desc}")

        # 显示维修项目（含工时）
        if not time_records.empty:
            print(f"\n🔧 维修项目:")
            for _, time_row in time_records.head(5).iterrows():
                repair_name = time_row['REPAIR_NAME']
                repair_hours = time_row.get('REPAIR_HOURS', None)

                if pd.notna(repair_hours) and repair_hours > 0:
                    print(f"   • {repair_name} (工时: {repair_hours}h)")
                else:
                    print(f"   • {repair_name}")

        # 显示更换配件
        if not parts_records.empty:
            print(f"\n🔩 更换配件:")
            for _, parts_row in parts_records.head(5).iterrows():
                print(f"   • {parts_row['PARTS_NAME']}")

        # 显示 LLM 分析结果
        print(f"\n🤖 LLM 分析:")
        print(f"   事件类型: {row['Event_Type']}")
        print(f"   系统: {row['System']}")
        print(f"   严重程度: {row['Severity']}")

        # 显示推理过程
        reasoning = row.get('Reasoning', '')
        if reasoning and reasoning != 'ERROR':
            print(f"   💭 推理: {reasoning}")

        # 判断是否使用了工时信息
        if '工时' in reasoning or '小时' in reasoning:
            print(f"   ✅ 已使用工时信息辅助判定")
        else:
            print(f"   ⚠️  未明确使用工时信息")

    # 6. 统计信息
    print(f"\n{'='*100}")
    print("【统计摘要】")
    print(f"{'='*100}\n")

    print(f"严重程度分布:")
    print(df_llm['Severity'].value_counts())
    print(f"\n事件类型分布:")
    print(df_llm['Event_Type'].value_counts())
    print(f"\n系统分布:")
    print(df_llm['System'].value_counts())

    # 7. 检查推理质量
    reasoning_with_hours = df_llm['Reasoning'].str.contains('工时|小时', na=False).sum()
    print(f"\n{'='*100}")
    print("【推理质量分析】")
    print(f"{'='*100}\n")
    print(f"总记录数: {len(df_llm)}")
    print(f"明确使用工时推理的记录: {reasoning_with_hours} ({reasoning_with_hours/len(df_llm)*100:.1f}%)")

    if reasoning_with_hours < len(df_llm) * 0.5:
        print(f"\n⚠️  注意: 不足 50% 的记录明确使用了工时信息")
        print(f"   这表明当前的 LLM 结果可能未启用工时辅助判定规则")
        print(f"\n💡 建议: 重新运行 LLM 处理以应用新的工时辅助判定规则")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
