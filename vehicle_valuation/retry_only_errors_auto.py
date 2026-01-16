#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接重试 ERROR 记录（自动确认版本）
"""

import sys
from pathlib import Path
import pandas as pd
import time
import os

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_single_record

def main():
    print("\n" + "="*80)
    print("直接重试 ERROR 记录")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 加载已有结果
    results_path = data_dir / "llm_parsed_results.csv"
    df_results = pd.read_csv(results_path)

    # 3. 找出所有 ERROR 记录
    error_mask = df_results['Severity'] == 'ERROR'
    error_ids = df_results.loc[error_mask, 'ID'].tolist()

    print(f"总记录数: {len(df_results)}")
    print(f"错误记录数: {len(error_ids)}\n")

    if len(error_ids) == 0:
        print("✓ 没有错误记录需要处理")
        return

    # 4. 获取 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('DEEPSEEK_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        os.environ['DEEPSEEK_API_KEY'] = api_key
                        break

    if not api_key:
        print("\n⚠️  未找到 DEEPSEEK_API_KEY")
        return

    # 5. 计算统计特征
    valid_hours = df_time[df_time['REPAIR_HOURS'] > 0]['REPAIR_HOURS']
    p50 = valid_hours.quantile(0.50)
    p85 = valid_hours.quantile(0.85)
    extreme_ratio = (valid_hours > 100).sum() / len(valid_hours) * 100

    system_prompt = f"""你是一位依据中国国家标准 (GB/T) 进行执业的二手车鉴定评估师。

【数据集统计背景 - 请务必参考】
本批次车辆维修数据的工时统计特征如下：
- 中位数 (P50): {p50:.1f} 小时 (意味着 50% 的维修在 {p50:.1f} 小时内完成)
- 高分位 (P85): 约 {p85:.1f} 小时 (意味着超过 {p85:.1f} 小时的维修仅占 15%)
- 极值 (>100h): 仅占 {extreme_ratio:.1f}%，可能是重大事故修复，也可能是数据录入错误

【双重验证判定逻辑】
请结合 "GB/T标准 (定性)" 和 "工时数据 (定量)" 进行综合判定：

1. **GB/T 标准定性**:
   - **L3 (重大)**: 事故车/总成大修 (GB/T 30323/5624)
   - **L2 (一般)**: 总成小修/重要部件更换
   - **L1 (轻微)**: 易损件/外观小修
   - **L0 (保养)**: 一级/二级维护 (GB/T 18344)

2. **工时定量修正 (基于统计特征)**:
   - **< {p50:.1f}h (快速)**: 即使涉及核心词(如"发动机检查")，也应降级为 L1
   - **{p50:.1f}h - {p85:.1f}h (常规)**: 典型的更换部件或总成小修 (L2)
   - **> {p85:.1f}h (显著)**: 强力支撑 L3 (大修) 的判定
   - **> 100h (异常)**: 请警惕！如果维修内容仅为简单项目(如换油)，请判定为数据错误并降级；只有内容确实涉及"全车翻新/严重事故"时才判定 L3

请输出 JSON 格式，推理必须包含: 1.理论定级(依据GB/T); 2.工时验证(参考统计分布)"""

    # 6. 自动开始处理（不需要确认）
    print("="*80)
    print(f"步骤 2: 自动重试 {len(error_ids)} 条错误记录")
    print("="*80 + "\n")

    max_retries = 3
    delay = 2  # 2秒延迟

    print(f"配置:")
    print(f"  - 重试次数: {max_retries}")
    print(f"  - 请求间隔: {delay} 秒")
    print(f"  - 预计耗时: 约 {len(error_ids) * delay / 60:.1f} 分钟\n")
    print(f"开始处理...\n")

    # 7. 逐条处理
    success_count = 0
    still_error_count = 0
    start_time = time.time()

    for i, error_id in enumerate(error_ids):
        print(f"[{i+1}/{len(error_ids):3d}] ID: {error_id[:10]}...", end=' ')

        try:
            result = process_single_record(
                error_id, df_base, df_parts, df_time,
                api_key, system_prompt, max_retries
            )

            if result['Severity'] != 'ERROR':
                print(f"✓ {result['Severity']:4s}")
                success_count += 1
                # 更新结果
                df_results.loc[df_results['ID'] == error_id, ['Event_Type', 'System', 'Severity', 'Reasoning']] = \
                    [result['Event_Type'], result['System'], result['Severity'], result['Reasoning']]
            else:
                print(f"✗ ERROR")
                still_error_count += 1

        except Exception as e:
            print(f"✗ EXC: {str(e)[:30]}")
            still_error_count += 1

        # 延迟
        if i < len(error_ids) - 1:
            time.sleep(delay)

        # 每50条保存一次
        if (i + 1) % 50 == 0:
            df_results.to_csv(results_path, index=False, encoding='utf-8-sig')
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  → 已保存 {i+1} 条 | 速度: {speed:.2f} 条/秒 | 预计剩余: {(len(error_ids)-i-1)/speed/60:.1f} 分钟")

    # 8. 最终保存
    df_results.to_csv(results_path, index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("处理完成")
    print("="*80)
    print(f"\n统计:")
    print(f"  - 成功修复: {success_count} 条")
    print(f"  - 仍然失败: {still_error_count} 条")
    print(f"  - 成功率: {success_count/len(error_ids)*100:.1f}%")
    print(f"  - 总耗时: {elapsed/60:.1f} 分钟")

    # 显示最终分布
    print(f"\n最终严重程度分布:")
    print(df_results['Severity'].value_counts())


if __name__ == "__main__":
    main()
