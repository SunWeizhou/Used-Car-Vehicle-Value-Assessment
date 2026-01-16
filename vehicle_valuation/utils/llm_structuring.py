#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 结构化处理模块

功能:
- 使用 DeepSeek API 对维修记录进行结构化分析
- 提取事件类型、受损系统、严重程度等关键信息
"""

import pandas as pd
import json
import re
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import signal
import sys


def process_sample_batch(base_df, parts_df, time_df, api_key, sample_size=200):
    """
    使用 LLM 处理维修记录样本，提取结构化信息

    Parameters:
    -----------
    base_df : pd.DataFrame
        基础信息表 (清洗后)
    parts_df : pd.DataFrame
        配件信息表 (清洗后)
    time_df : pd.DataFrame
        工时信息表 (清洗后)
    api_key : str
        DeepSeek API 密钥
    sample_size : int
        处理样本数量 (默认: 200)

    Returns:
    --------
    results_df : pd.DataFrame
        包含 ID 和提取的结构化信息的 DataFrame
    """
    # 初始化 DeepSeek 客户端
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key
    )

    # 计算工时统计特征（用于 System Prompt 背景信息）
    valid_hours = time_df[time_df['REPAIR_HOURS'] > 0]['REPAIR_HOURS']
    p50 = valid_hours.quantile(0.50)
    p85 = valid_hours.quantile(0.85)
    extreme_ratio = (valid_hours > 100).sum() / len(valid_hours) * 100

    # 构建带有统计背景的 System Prompt
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

    # 抽取样本
    sample_df = base_df.head(sample_size).copy()
    print(f"开始处理 {len(sample_df)} 条样本记录...\n")

    # 存储结果
    results = []
    success_count = 0
    error_count = 0

    # 遍历每一行样本
    for idx, row in sample_df.iterrows():
        record_id = row['ID']

        try:
            # 1. 信息聚合：根据 ID 匹配配件和工时信息
            parts_list = parts_df[parts_df['RECORD_ID'] == record_id]['PARTS_NAME'].tolist()

            # 提取工时信息，包含工时数
            time_records = time_df[time_df['RECORD_ID'] == record_id]
            time_list_with_hours = []
            for _, time_row in time_records.iterrows():
                repair_name = time_row['REPAIR_NAME']
                repair_hours = time_row.get('REPAIR_HOURS', None)

                # 如果工时有效且大于0，则显示工时
                if pd.notna(repair_hours) and repair_hours > 0:
                    time_list_with_hours.append(f"{repair_name} (工时: {repair_hours}h)")
                else:
                    # 工时缺失或为0，只显示项目名称
                    time_list_with_hours.append(repair_name)

            # 2. 构建 Context
            fault_desc = row.get('FAULT_DESC', '无').strip() if pd.notna(row.get('FAULT_DESC')) else '无'

            context_parts = []
            context_parts.append(f"故障描述: {fault_desc}")
            if time_list_with_hours:
                context_parts.append(f"维修项目: {', '.join(time_list_with_hours[:10])}")  # 限制数量
            if parts_list:
                context_parts.append(f"更换配件: {', '.join(parts_list[:10])}")  # 限制数量

            context = "\n".join(context_parts)

            # 3. 构造 User Prompt (简洁版，统计背景已在 System Prompt 中)
            prompt = f"""请分析以下维修记录并提取结构化信息。

维修记录:
{context}

请以 JSON 格式输出：
{{
  "Event_Type": "保养" 或 "维修",
  "System": "发动机" | "变速箱" | "车身" | "底盘" | "电气" | "常规耗材",
  "Severity": "L0" | "L1" | "L2" | "L3",
  "Reasoning": "必须严格遵循双重验证格式：'依据 [具体国标条款]，[维修对象]属于[理论等级]；但[工时数据][分析结论]，故[最终判定]'。例如：'依据 GB/T 5624，制动软管属于安全件(理论 L2)；但工时仅 0.7h，符合快速更换特征，故综合判定为 L1' 或 '依据 GB/T 30323，大梁校正属于事故车修复(理论 L3)；且工时高达 15h，确认涉及结构性操作，维持 L3 判定'"
}}

只返回 JSON，不要其他内容。"""

            # 4. API 调用
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # 5. 解析返回内容
            content = response.choices[0].message.content.strip()

            # 移除可能的 Markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

            # 解析 JSON
            parsed = json.loads(content)

            # 6. 保存结果
            reasoning = parsed.get('Reasoning', '')
            results.append({
                'ID': record_id,
                'Event_Type': parsed.get('Event_Type', '未知'),
                'System': parsed.get('System', '未知'),
                'Severity': parsed.get('Severity', '未知'),
                'Reasoning': reasoning
            })

            # 打印推理过程 (用于验证)
            print(f"✓ 记录 {record_id}: {parsed.get('Severity', '未知')} - {reasoning}")

            success_count += 1

        except json.JSONDecodeError as e:
            print(f"✗ 记录 {record_id}: JSON 解析失败 - {str(e)[:50]}")
            # 保存失败的记录，标记为错误
            results.append({
                'ID': record_id,
                'Event_Type': 'ERROR',
                'System': 'ERROR',
                'Severity': 'ERROR',
                'Reasoning': f'JSON 解析失败: {str(e)[:100]}'
            })
            error_count += 1

        except Exception as e:
            print(f"✗ 记录 {record_id}: 处理失败 - {str(e)[:50]}")
            # 保存失败的记录，标记为错误
            results.append({
                'ID': record_id,
                'Event_Type': 'ERROR',
                'System': 'ERROR',
                'Severity': 'ERROR',
                'Reasoning': f'处理失败: {str(e)[:100]}'
            })
            error_count += 1

        # 7. 进度反馈
        current_total = success_count + error_count
        if current_total % 10 == 0:
            print(f"已处理: {current_total}/{sample_size} (成功: {success_count}, 失败: {error_count})...")

    # 8. 转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 9. 保存结果
    output_path = Path("vehicle_valuation/data/llm_parsed_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n处理完成！")
    print(f"总计: {len(results_df)} 条记录")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条")
    print(f"成功率: {success_count/len(results_df)*100:.2f}%")
    print(f"结果已保存至: {output_path}")

    # 打印统计信息
    if success_count > 0:
        print("\n【统计摘要】")
        valid_df = results_df[results_df['Severity'] != 'ERROR']
        print(f"事件类型分布:")
        print(valid_df['Event_Type'].value_counts())
        print(f"\n严重程度分布:")
        print(valid_df['Severity'].value_counts())
        print(f"\n系统分布:")
        print(valid_df['System'].value_counts())

    return results_df


def test_llm_connection(api_key):
    """
    测试 DeepSeek API 连接

    Parameters:
    -----------
    api_key : str
        DeepSeek API 密钥
    """
    try:
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "你好，请回复'连接成功'"}
            ],
            max_tokens=50
        )

        print("✓ API 连接成功")
        print(f"响应: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"✗ API 连接失败: {str(e)}")
        return False


def process_single_record(record_id, base_df, parts_df, time_df, api_key, system_prompt, max_retries=3):
    """
    处理单条记录的函数（用于并发调用）
    支持自动重试机制

    Parameters:
    -----------
    max_retries : int
        最大重试次数（默认: 3）

    Returns:
    --------
    dict or None
        成功返回结果字典，失败返回 ERROR 字典
    """
    # 初始化客户端（每个线程独立）
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key
    )

    # 重试循环
    for attempt in range(max_retries):
        try:
            # 获取记录
            row = base_df[base_df['ID'] == record_id].iloc[0]

            # 信息聚合（添加编码安全处理）
            parts_data = parts_df[parts_df['RECORD_ID'] == record_id]['PARTS_NAME']
            parts_list = []
            for name in parts_data:
                if pd.notna(name):
                    try:
                        # 确保字符串是有效的 UTF-8
                        if isinstance(name, str):
                            parts_list.append(name.encode('utf-8', errors='ignore').decode('utf-8'))
                        else:
                            parts_list.append(str(name))
                    except:
                        parts_list.append(str(name))

            time_records = time_df[time_df['RECORD_ID'] == record_id]
            time_list_with_hours = []
            for _, time_row in time_records.iterrows():
                repair_name = time_row.get('REPAIR_NAME', '')
                repair_hours = time_row.get('REPAIR_HOURS', None)

                # 安全处理 repair_name
                if pd.isna(repair_name) or repair_name == '':
                    repair_name = '未知维修项目'
                else:
                    try:
                        repair_name = str(repair_name).encode('utf-8', errors='ignore').decode('utf-8')
                    except:
                        repair_name = str(repair_name)

                if pd.notna(repair_hours) and repair_hours > 0:
                    time_list_with_hours.append(f"{repair_name} (工时: {repair_hours}h)")
                else:
                    time_list_with_hours.append(repair_name)

            # 构建上下文（添加编码安全处理）
            fault_desc = row.get('FAULT_DESC', '无')
            if pd.notna(fault_desc):
                try:
                    fault_desc = str(fault_desc).encode('utf-8', errors='ignore').decode('utf-8').strip()
                except:
                    fault_desc = str(fault_desc).strip()
            else:
                fault_desc = '无'

            context_parts = []
            context_parts.append(f"故障描述: {fault_desc}")
            if time_list_with_hours:
                context_parts.append(f"维修项目: {', '.join(time_list_with_hours[:10])}")
            if parts_list:
                context_parts.append(f"更换配件: {', '.join(parts_list[:10])}")

            # 安全地拼接上下文（使用 errors='ignore' 避免编码错误）
            context = "\n".join([part.encode('utf-8', errors='ignore').decode('utf-8') for part in context_parts])

            # 构造 prompt
            prompt = f"""请分析以下维修记录并提取结构化信息。

维修记录:
{context}

请以 JSON 格式输出：
{{
  "Event_Type": "保养" 或 "维修",
  "System": "发动机" | "变速箱" | "车身" | "底盘" | "电气" | "常规耗材",
  "Severity": "L0" | "L1" | "L2" | "L3",
  "Reasoning": "必须严格遵循双重验证格式：'依据 [具体国标条款]，[维修对象]属于[理论等级]；但[工时数据][分析结论]，故[最终判定]'。例如：'依据 GB/T 5624，制动软管属于安全件(理论 L2)；但工时仅 0.7h，符合快速更换特征，故综合判定为 L1' 或 '依据 GB/T 30323，大梁校正属于事故车修复(理论 L3)；且工时高达 15h，确认涉及结构性操作，维持 L3 判定'"
}}

只返回 JSON，不要其他内容。"""

            # API 调用（增加超时设置）
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=30.0  # 30秒超时
            )

            # 解析返回内容
            content = response.choices[0].message.content.strip()

            # 移除可能的 Markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 解析 JSON
            parsed = json.loads(content)

            # 成功则返回
            return {
                'ID': record_id,
                'Event_Type': parsed.get('Event_Type', '未知'),
                'System': parsed.get('System', '未知'),
                'Severity': parsed.get('Severity', '未知'),
                'Reasoning': parsed.get('Reasoning', '')
            }

        except Exception as e:
            # 如果是最后一次尝试，则返回错误
            if attempt == max_retries - 1:
                return {
                    'ID': record_id,
                    'Event_Type': 'ERROR',
                    'System': 'ERROR',
                    'Severity': 'ERROR',
                    'Reasoning': f'处理失败: {str(e)[:100]}'
                }
            # 否则等待一段时间后重试（指数退避）
            else:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue

    # 理论上不会到达这里
    return {
        'ID': record_id,
        'Event_Type': 'ERROR',
        'System': 'ERROR',
        'Severity': 'ERROR',
        'Reasoning': '处理失败: 未知原因'
    }


def process_sample_batch_concurrent(base_df, parts_df, time_df, api_key, sample_size=200, max_workers=5, max_retries=3, request_delay=0.2):
    """
    并发处理维修记录样本，提取结构化信息（大幅提升处理速度）

    Parameters:
    -----------
    base_df : pd.DataFrame
        基础信息表 (清洗后)
    parts_df : pd.DataFrame
        配件信息表 (清洗后)
    time_df : pd.DataFrame
        工时信息表 (清洗后)
    api_key : str
        DeepSeek API 密钥
    sample_size : int
        处理样本数量 (默认: 200)
    max_workers : int
        并发线程数 (默认: 5，已根据 API 限制优化，不建议超过 10)
    max_retries : int
        失败重试次数 (默认: 3，使用指数退避策略)
    request_delay : float
        每次请求后的延迟秒数 (默认: 0.2秒，用于避免 API 速率限制)

    Returns:
    --------
    results_df : pd.DataFrame
        包含 ID 和提取的结构化信息的 DataFrame
    """
    # 计算工时统计特征
    valid_hours = time_df[time_df['REPAIR_HOURS'] > 0]['REPAIR_HOURS']
    p50 = valid_hours.quantile(0.50)
    p85 = valid_hours.quantile(0.85)
    extreme_ratio = (valid_hours > 100).sum() / len(valid_hours) * 100

    # 构建带有统计背景的 System Prompt
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

    # 抽取样本
    sample_df = base_df.head(sample_size).copy()
    record_ids = sample_df['ID'].tolist()

    print(f"开始并发处理 {len(record_ids)} 条样本记录...")
    print(f"并发线程数: {max_workers}\n")

    # 存储结果
    results = []
    success_count = 0
    error_count = 0
    lock = Lock()

    # 用于增量保存
    output_path = Path(__file__).parent.parent / "data" / "llm_parsed_results.csv"
    start_time = time.time()

    # 全局变量用于信号处理
    interrupt_flag = False

    def signal_handler(signum, frame):
        nonlocal interrupt_flag
        print("\n\n⚠️  收到中断信号，正在保存已处理的数据...")
        interrupt_flag = True

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查是否存在已处理的结果（断点续传）
    processed_ids = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df['ID'].astype(str).tolist())
        print(f"✓ 发现已处理的 {len(processed_ids)} 条记录，将跳过这些记录")

        # 过滤掉已处理的记录
        record_ids = [rid for rid in record_ids if str(rid) not in processed_ids]
        if not record_ids:
            print(f"\n所有记录已处理完成！")
            return existing_df

        print(f"✓ 剩余 {len(record_ids)} 条记录待处理\n")

    # 使用线程池并发处理
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务（传递 max_retries 参数）
            future_to_id = {
                executor.submit(process_single_record, record_id, base_df, parts_df, time_df, api_key, system_prompt, max_retries): record_id
                for record_id in record_ids
            }

            # 处理完成的任务
            for future in as_completed(future_to_id):
                # 检查是否收到中断信号
                if interrupt_flag:
                    print("\n正在中断处理...")
                    # 取消所有未完成的任务
                    for f in future_to_id:
                        f.cancel()
                    break

                record_id = future_to_id[future]

                try:
                    result = future.result()
                    results.append(result)

                    # 添加请求延迟（避免 API 速率限制）
                    if request_delay > 0:
                        time.sleep(request_delay)

                    with lock:
                        if result['Severity'] != 'ERROR':
                            success_count += 1
                        else:
                            error_count += 1

                        # 实时打印进度
                        current_total = success_count + error_count
                        if current_total % 10 == 0:
                            elapsed = time.time() - start_time
                            speed = current_total / elapsed if elapsed > 0 else 0
                            remaining = (len(record_ids) - current_total) / speed if speed > 0 else 0
                            print(f"已处理: {current_total}/{len(record_ids)} (成功: {success_count}, 失败: {error_count}) | 速度: {speed:.1f} 条/秒 | 预计剩余: {remaining/60:.1f} 分钟")

                        # 增量保存（每50条保存一次，更安全）
                        if current_total % 50 == 0:
                            # 合并已有结果和新结果
                            if output_path.exists():
                                existing_df = pd.read_csv(output_path)
                                combined_results = existing_df.to_dict('records') + results
                            else:
                                combined_results = results

                            temp_df = pd.DataFrame(combined_results)
                            temp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                            print(f"✓ 已增量保存 {len(combined_results)} 条结果")

                except Exception as e:
                    with lock:
                        error_count += 1
                        results.append({
                            'ID': record_id,
                            'Event_Type': 'ERROR',
                            'System': 'ERROR',
                            'Severity': 'ERROR',
                            'Reasoning': f'线程处理异常: {str(e)[:100]}'
                        })

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在保存已处理的数据...")
        if interrupt_flag:
            raise  # 重新抛出以被外层 except 捕获
    except Exception as e:
        print(f"\n\n✗ 处理过程发生错误: {str(e)}")
        print(f"正在保存已处理的 {len(results)} 条结果...")

    finally:
        # 无论如何都要保存已处理的结果
        if results:
            # 合并已有结果
            if output_path.exists():
                try:
                    existing_df = pd.read_csv(output_path)
                    # 去重（保留最新的）
                    all_results = existing_df.to_dict('records') + results
                    # 按 ID 去重，保留后面的
                    unique_results = {}
                    for r in all_results:
                        unique_results[r['ID']] = r
                    results = list(unique_results.values())
                except Exception as e:
                    print(f"⚠️  读取已有结果失败: {e}，仅保存新结果")

            # 转换为 DataFrame 并保存
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            if interrupt_flag:
                print(f"\n✓ 已保存 {len(results_df)} 条结果到: {output_path}")
                print(f"✓ 程序已安全中断，下次运行将从断点继续")
                sys.exit(0)

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)

    # 最终保存（合并已有结果）
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # 去重
        all_results = existing_df.to_dict('records') + results
        unique_results = {}
        for r in all_results:
            unique_results[r['ID']] = r
        results_df = pd.DataFrame(list(unique_results.values()))

    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time
    print(f"\n处理完成！")
    print(f"总计: {len(results_df)} 条记录")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条")
    print(f"成功率: {success_count/len(results_df)*100:.2f}%")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"平均速度: {len(results_df)/elapsed:.1f} 条/秒")
    print(f"结果已保存至: {output_path}")

    # 打印统计信息
    if success_count > 0:
        print("\n【统计摘要】")
        valid_df = results_df[results_df['Severity'] != 'ERROR']
        print(f"事件类型分布:")
        print(valid_df['Event_Type'].value_counts())
        print(f"\n严重程度分布:")
        print(valid_df['Severity'].value_counts())
        print(f"\n系统分布:")
        print(valid_df['System'].value_counts())

    return results_df


if __name__ == "__main__":
    # 测试代码示例
    print("LLM 结构化处理模块")
    print("="*80)
    print("\n使用方法:")
    print("  from utils.llm_structuring import process_sample_batch")
    print("  results_df = process_sample_batch(df_base, df_parts, df_time, api_key, sample_size=200)")
    print("\n或测试 API 连接:")
    print("  from utils.llm_structuring import test_llm_connection")
    print("  test_llm_connection('your-api-key')")
