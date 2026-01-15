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

            # 3. 构造 Prompt (基于中国国家标准 GB/T + 工时辅助判定)
            prompt = f"""你是一位依据中国国家标准 (GB/T) 进行执业的二手车鉴定评估师。请分析以下商用车的维修记录，并提取结构化信息。

维修记录:
{context}

【双重验证机制 (Dual-Factor Validation)】
你必须采用"国标为主、工时为辅"的双重验证逻辑进行判定：

**第一步 (定性)**: 依据中国国家标准 (GB/T) 确定维修对象的理论等级
- **L3 (重大/Major)** - 依据 GB/T 30323-2013 和 GB/T 5624-2005:
  * 涉及 "事故车" 判定标准（车架/大梁矫正、气囊弹出、火烧/水泡）
  * 涉及 "总成大修"（发动机、变速箱的解体维修或总成更换）
  * 涉及核心安全系统（制动失灵、转向失效）导致的重大部件更换

- **L2 (一般/Moderate)**:
  * 涉及 "总成小修" 或次要总成更换（发电机、起动机、空调压缩机、水箱、半轴）
  * 虽导致车辆停驶，但未伤及发动机/变速箱/车架核心结构

- **L1 (轻微/Minor)** - 依据 GB/T 5624 "汽车小修":
  * 仅涉及易损件更换（轮胎、蓄电池、刹车片、灯泡、雨刮）
  * 外观覆盖件修复（喷漆、钣金）但不涉及结构件
  * 舒适性配置维修（收音机、座椅、玻璃升降）

- **L0 (保养/Maintenance)** - 依据 GB/T 18344-2016:
  * 属于 "一级/二级维护" 范畴
  * 包含：更换机油、三滤、油液添加、清洗积碳、四轮定位、例行检查

**第二步 (定量)**: 使用工时数据对理论等级进行验证或修正
- **验证逻辑**: 工时必须与国标判定的理论等级相匹配
- **修正规则**:
  * 若理论等级为 L3 但工时 <1h → 说明并非总成解体，降级为 L1
  * 若理论等级为 L1/L2 但工时 >8h 且为机械拆装 → 可能低估，升级为 L3
  * 若理论等级为 L3 且工时 >8h → 确认重大维修，维持 L3
  * 高工时的喷漆/钣金维持 L1/L2 (车身维修)，不升级为 L3

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
                    {"role": "system", "content": "你是专业的二手车鉴定评估师，严格遵循中国国家标准 (GB/T) 进行故障等级划分。你必须采用【双重验证机制】：第一步依据 GB/T 标准确定理论等级（定性为主），第二步使用工时数据进行验证或修正（定量为辅）。推理过程必须清晰展示'国标判定+工时验证'两层逻辑，不得仅依赖工时数据。"},
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
