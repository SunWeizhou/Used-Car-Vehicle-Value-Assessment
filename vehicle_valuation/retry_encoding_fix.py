#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry failed records with encoding fix
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_single_record
import os
import time

def safe_str(obj):
    """Safely convert object to string, handling encoding issues"""
    if pd.isna(obj):
        return ''
    if isinstance(obj, float):
        if np.isnan(obj):
            return ''
        return str(obj)
    try:
        return str(obj).encode('utf-8', errors='ignore').decode('utf-8')
    except:
        return str(obj)

def main():
    print("\n" + "="*80)
    print("重试失败记录 - 编码安全版本")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 加载已有结果
    results_path = data_dir / "llm_parsed_results.csv"
    existing_results = pd.read_csv(results_path)
    print(f"✓ 已有结果: {len(existing_results)} 条记录\n")

    # 3. 筛选失败的记录
    failed_records = existing_results[existing_results['Severity'] == 'ERROR']
    print(f"步骤 2: 发现失败记录 {len(failed_records)} 条\n")

    if len(failed_records) == 0:
        print("✓ 没有需要重试的记录")
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

    # 6. 配置
    print("="*80)
    print(f"步骤 3: 重试失败记录（共 {len(failed_records)} 条）")
    print("="*80 + "\n")

    batch_size = 50
    max_retries = 3
    request_delay = 2  # 2秒延迟

    print(f"⚠️  配置:")
    print(f"  - 批量大小: {batch_size} 条/批")
    print(f"  - 重试次数: {max_retries} 次")
    print(f"  - 请求间隔: {request_delay} 秒")
    print(f"预计批数: {(len(failed_records) + batch_size - 1) // batch_size} 批\n")

    confirm = input("是否继续？(输入 'yes' 确认): ").strip()
    if confirm.lower() != 'yes':
        print("\n处理已取消")
        return

    # 7. 获取需要重试的 ID
    retry_ids = failed_records['ID'].tolist()
    retry_df = df_base[df_base['ID'].isin(retry_ids)].copy()

    print(f"\n开始重试 {len(retry_df)} 条失败记录...\n")

    # 8. 分批处理
    all_results = []
    success_count = 0
    error_count = 0
    start_time = time.time()

    total_batches = (len(retry_df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(retry_df))
        batch_ids = retry_df['ID'].iloc[start_idx:end_idx].tolist()

        print(f"\n{'='*60}")
        print(f"批次 {batch_idx + 1}/{total_batches}: 处理 {len(batch_ids)} 条记录")
        print(f"{'='*60}\n")

        batch_results = []
        for idx, record_id in enumerate(batch_ids):
            print(f"  [{idx+1}/{len(batch_ids)}] 处理 ID: {record_id[:8]}...", end=' ')

            try:
                result = process_single_record(
                    record_id, df_base, df_parts, df_time,
                    api_key, system_prompt, max_retries
                )

                if result['Severity'] != 'ERROR':
                    print(f"✓ {result['Severity']}")
                    success_count += 1
                else:
                    print(f"✗ ERROR")
                    error_count += 1

                batch_results.append(result)
                time.sleep(request_delay)

            except Exception as e:
                print(f"✗ EXCEPTION: {str(e)[:50]}")
                error_count += 1
                batch_results.append({
                    'ID': record_id,
                    'Event_Type': 'ERROR',
                    'System': 'ERROR',
                    'Severity': 'ERROR',
                    'Reasoning': f'Batch processing error: {str(e)[:100]}'
                })

        all_results.extend(batch_results)

        # 显示进度
        current_total = success_count + error_count
        elapsed = time.time() - start_time
        speed = current_total / elapsed if elapsed > 0 else 0
        print(f"\n  批次完成: 成功 {success_count - (current_total - len(batch_results))} / 失败 {error_count - (current_total - len(batch_results))}")
        print(f"  总进度: {current_total}/{len(retry_df)} (成功: {success_count}, 失败: {error_count})")
        print(f"  速度: {speed:.2f} 条/秒")

        # 每批之后保存中间结果
        if batch_idx < total_batches - 1:
            print(f"\n  等待 {request_delay * 2} 秒后继续...")
            time.sleep(request_delay * 2)

    # 9. 合并结果
    print("\n" + "="*80)
    print("步骤 4: 合并结果")
    print("="*80 + "\n")

    # 删除旧的失败记录
    success_records = existing_results[existing_results['Severity'] != 'ERROR'].copy()

    # 合并成功记录和新结果
    all_results_df = pd.DataFrame(all_results)
    combined = pd.concat([success_records, all_results_df], ignore_index=True)

    # 按 ID 去重
    combined = combined.drop_duplicates(subset=['ID'], keep='last')

    # 保存
    combined.to_csv(results_path, index=False, encoding='utf-8-sig')

    print(f"\n✓ 合并完成")
    print(f"  - 原成功记录: {len(success_records)} 条")
    print(f"  - 新处理记录: {len(all_results_df)} 条")
    print(f"  - 合并后总计: {len(combined)} 条")
    print(f"  - 结果已保存至: {results_path}\n")

    # 统计
    print("【最终统计】")
    print(f"\n严重程度分布:")
    print(combined['Severity'].value_counts())
    print(f"\n事件类型分布:")
    print(combined['Event_Type'].value_counts())

    errors = combined[combined['Severity'] == 'ERROR']
    if len(errors) > 0:
        print(f"\n仍有失败记录: {len(errors)} 条 ({len(errors)/len(combined)*100:.1f}%)")
    else:
        print("\n✓ 所有记录处理成功！")

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print(f"平均速度: {len(retry_df)/elapsed:.2f} 条/秒")


if __name__ == "__main__":
    main()
