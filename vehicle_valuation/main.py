#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车残值评估系统 - 主入口

功能:
- 数据加载与预处理
- 车辆生命周期分析
- 使用强度与保养规范度评估
- 故障率建模
- 未来风险预测
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data, print_data_summary
from models.lifecycle import prepare_weibull_data, WeibullModel
from models.behavior import BehaviorModel
from models.reliability import ReliabilityModel
from models.weighting import PCAWeightingModel
import numpy as np
import pandas as pd


def main():
    """主函数"""
    print("\n" + "="*80)
    print("二手车残值评估系统")
    print("="*80 + "\n")

    # 1. 加载和清洗数据
    print("步骤 1: 加载和清洗数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 数据验证统计
    print("\n" + "="*80)
    print("步骤 2: 数据验证统计")
    print("="*80)

    # 2.1 有效时间范围
    print("\n【有效时间范围】")
    print(f"  最小日期: {df_base['SETTLE_DATE'].min()}")
    print(f"  最大日期: {df_base['SETTLE_DATE'].max()}")
    print(f"  时间跨度: {(df_base['SETTLE_DATE'].max() - df_base['SETTLE_DATE'].min()).days} 天")

    # 2.2 有效里程范围
    print("\n【有效里程范围】")
    print(f"  最小里程: {df_base['REPAIR_MILEAGE'].min():,.0f} km")
    print(f"  最大里程: {df_base['REPAIR_MILEAGE'].max():,.0f} km")
    print(f"  平均里程: {df_base['REPAIR_MILEAGE'].mean():,.0f} km")
    print(f"  中位里程: {df_base['REPAIR_MILEAGE'].median():,.0f} km")

    # 2.3 剩余数据量
    print("\n【剩余数据量】")
    print(f"  baseinfo:    {len(df_base):,} 条维修记录")
    print(f"  parts_info:  {len(df_parts):,} 条配件记录")
    print(f"  time_info:   {len(df_time):,} 条工时记录")

    # 2.4 内存占用情况
    print("\n【内存占用情况】")
    base_memory = df_base.memory_usage(deep=True).sum() / 1024**2
    parts_memory = df_parts.memory_usage(deep=True).sum() / 1024**2
    time_memory = df_time.memory_usage(deep=True).sum() / 1024**2
    total_memory = base_memory + parts_memory + time_memory

    print(f"  baseinfo:    {base_memory:8.2f} MB")
    print(f"  parts_info:  {parts_memory:8.2f} MB")
    print(f"  time_info:   {time_memory:8.2f} MB")
    print(f"  " + "-"*40)
    print(f"  总计:        {total_memory:8.2f} MB")

    # 2.5 数据质量检查
    print("\n【数据质量检查】")
    print(f"  ✓ ID 列类型: {df_base['ID'].dtype}")
    print(f"  ✓ 日期列类型: {df_base['SETTLE_DATE'].dtype}")
    print(f"  ✓ 里程列类型: {df_base['REPAIR_MILEAGE'].dtype}")
    print(f"  ✓ 配件 RECORD_ID 类型: {df_parts['RECORD_ID'].dtype}")
    print(f"  ✓ 工时 RECORD_ID 类型: {df_time['RECORD_ID'].dtype}")

    print("\n" + "="*80)
    print("✓ 数据清洗验证完成！数据质量良好，可以进行后续分析。")
    print("="*80 + "\n")

    # 3. Weibull 生命周期建模
    print("\n" + "="*80)
    print("步骤 3: Weibull 生命周期建模")
    print("="*80)

    # 3.1 准备 Weibull 数据
    print("\n【数据准备】")
    weibull_df = prepare_weibull_data(df_base)
    print(f"  车辆数量: {len(weibull_df):,}")
    print(f"  已失效车辆 (event=1): {weibull_df['event'].sum():,}")
    print(f"  存活车辆 (event=0, 右截断): {(weibull_df['event'] == 0).sum():,}")

    # 3.2 拟合 Weibull 模型
    print("\n【模型拟合】")
    model = WeibullModel()
    model.fit(
        t=weibull_df['t'].values,
        event=weibull_df['event'].values
    )

    # 3.3 输出参数
    params = model.get_params()
    print(f"\n【拟合参数】")
    print(f"  形状参数 k:  {params['k']:.4f}")
    print(f"  尺度参数 λ:  {params['lambda_']:,.0f} km")
    print(f"\n参数解释:")
    print(f"  - k < 1: 故障率随时间下降 (早期失效)")
    print(f"  - k = 1: 故障率恒定 (随机失效, 指数分布)")
    print(f"  - k > 1: 故障率随时间上升 (磨损失效)")

    # 3.4 案例展示
    print("\n【案例展示 - 随机 5 辆车】")
    np.random.seed(42)
    sample_vins = weibull_df.sample(5)

    for idx, row in sample_vins.iterrows():
        vin = row['VIN']
        t_current = row['t']
        event = int(row['event'])
        event_label = "已报废" if event == 1 else "存活 (右截断)"

        score = model.predict_score(t_current)

        print(f"\n车辆 {vin[:8]}...")
        print(f"  当前里程: {t_current:,.0f} km")
        print(f"  状态: {event_label}")
        print(f"  生命周期得分: {score:.2f} / 100")

    print("\n" + "="*80)
    print("✓ Weibull 生命周期建模完成！")
    print("="*80 + "\n")

    # 4. 行为模型 - 使用强度与保养规范度评估
    print("\n" + "="*80)
    print("步骤 4: ECDF 行为模型")
    print("="*80)

    # 4.1 尝试加载 LLM 结果
    llm_results_path = Path(__file__).parent / "data" / "llm_parsed_results.csv"
    df_llm = None
    if llm_results_path.exists():
        print(f"\n【加载 LLM 结果】")
        print(f"  找到 LLM 结果文件: {llm_results_path.name}")
        try:
            df_llm = pd.read_csv(llm_results_path)
            print(f"  LLM 结果记录数: {len(df_llm):,}")
        except Exception as e:
            print(f"  ⚠ 读取 LLM 结果失败: {e}")
            df_llm = None
    else:
        print(f"\n【使用关键词规则】")
        print(f"  未找到 LLM 结果文件，使用关键词规则识别保养")

    # 4.2 拟合行为模型
    print("\n【模型拟合】")
    behavior_model = BehaviorModel()
    behavior_model.fit(df_base, df_llm=df_llm)

    # 4.3 显示统计信息
    print(f"\n【数据统计】")
    print(f"  车辆数量: {behavior_model.stats['n_vehicles']:,}")

    print(f"\n【使用强度指标对比 (修正版 vs 原始版)】")
    print(f"  修正版 (里程增量法):")
    print(f"    - 均值: {behavior_model.stats['avg_daily_mileage_mean']:.2f} km/天")
    print(f"    - 中位数: {behavior_model.stats['avg_daily_mileage_median']:.2f} km/天")
    print(f"  原始版 (累积里程法):")
    print(f"    - 均值: {behavior_model.stats['avg_daily_mileage_original_mean']:.2f} km/天")
    print(f"    - 中位数: {behavior_model.stats['avg_daily_mileage_original_median']:.2f} km/天")
    print(f"  改进幅度: {(1 - behavior_model.stats['avg_daily_mileage_mean'] / behavior_model.stats['avg_daily_mileage_original_mean'])*100:.1f}%")

    print(f"\n【保养规范度】")
    print(f"  保养密度 - 均值: {behavior_model.stats['maint_density_mean']:.4f} 次/万公里")
    print(f"  保养密度 - 中位数: {behavior_model.stats['maint_density_median']:.4f} 次/万公里")

    # 4.4 准备车辆数据用于评分
    # 按 VIN 聚合获取每辆车的里程、天数、保养次数
    df_base_copy = df_base.copy()
    df_base_copy['SETTLE_DATE'] = pd.to_datetime(df_base_copy['SETTLE_DATE'])

    # 识别保养
    if df_llm is None:
        maintenance_keywords = ['保养', '更换机油', '机滤', '三滤', '润滑油']
        df_base_copy['is_maintenance'] = df_base_copy['FAULT_DESC'].str.contains(
            '|'.join(maintenance_keywords), na=False
        )
    else:
        df_base_copy = df_base_copy.merge(
            df_llm[['ID', 'Event_Type']],
            on='ID',
            how='left'
        )
        df_base_copy['is_maintenance'] = (df_base_copy['Event_Type'] == '保养').fillna(False)

    # 聚合
    vehicle_data = df_base_copy.groupby('VIN').agg({
        'REPAIR_MILEAGE': ['min', 'max'],
        'SETTLE_DATE': ['min', 'max'],
        'is_maintenance': 'sum'
    }).reset_index()
    vehicle_data.columns = ['VIN', 'min_mileage', 'max_mileage', 'first_date', 'last_date', 'maint_count']
    vehicle_data['span_days'] = (vehicle_data['last_date'] - vehicle_data['first_date']).dt.days
    vehicle_data.loc[vehicle_data['span_days'] < 30, 'span_days'] = 30

    # 4.5 案例展示（沿用之前的 5 辆车）
    print("\n【案例展示 - 同样的 5 辆车】")
    for idx, row in sample_vins.iterrows():
        vin = row['VIN']
        t_current = row['t']

        # 从 vehicle_data 获取信息
        veh_row = vehicle_data[vehicle_data['VIN'] == vin]
        if veh_row.empty:
            continue

        min_mileage = veh_row.iloc[0]['min_mileage']
        mileage = veh_row.iloc[0]['max_mileage']
        days = veh_row.iloc[0]['span_days']
        maint_count = int(veh_row.iloc[0]['maint_count'])

        # 预测得分（使用修正版：传入 min_mileage）
        usage_score, maint_score = behavior_model.predict_scores(mileage, days, maint_count, min_mileage=min_mileage)

        print(f"\n车辆 {vin[:8]}...")
        print(f"  总里程: {mileage:,.0f} km")
        print(f"  使用天数: {days} 天")
        print(f"  保养次数: {maint_count} 次")
        print(f"  使用强度得分: {usage_score:.2f} / 100 (越低越激烈)")
        print(f"  保养规范度得分: {maint_score:.2f} / 100 (越高越规范)")

    print("\n" + "="*80)
    print("✓ ECDF 行为模型建模完成！")
    print("="*80 + "\n")

    # 5. 可靠性模型 - 故障率强度评估
    print("\n" + "="*80)
    print("步骤 5: 故障率强度模型 (基于 LLM 标注)")
    print("="*80)

    # 5.1 检查 LLM 结果
    if df_llm is None:
        print("\n⚠ 警告: 未找到 LLM 结果文件，跳过可靠性模型训练")
        print("   提示: 运行 LLM 结构化处理生成 llm_parsed_results.csv")
    else:
        # 5.2 拟合可靠性模型
        print("\n【模型拟合】")
        reliability_model = ReliabilityModel()
        reliability_model.fit(llm_df=df_llm, base_df=df_base)

        # 5.3 显示统计信息
        print(f"\n【数据统计】")
        print(f"  LLM 记录数: {reliability_model.stats['n_llm_records']:,}")
        print(f"  覆盖车辆数: {reliability_model.stats['n_vehicles']:,}")
        print(f"  平均每车记录数: {reliability_model.stats['avg_records_per_vehicle']:.2f}")
        print(f"  故障率强度 (Λ) - 均值: {reliability_model.stats['lambda_mean']:.6f} /km")
        print(f"  故障率强度 (Λ) - 中位数: {reliability_model.stats['lambda_median']:.6f} /km")
        print(f"  故障率强度 (Λ) - 最小值: {reliability_model.stats['lambda_min']:.6f} /km")
        print(f"  故障率强度 (Λ) - 最大值: {reliability_model.stats['lambda_max']:.6f} /km")
        print(f"  群体基准 (Λ_pop): {reliability_model.lambda_pop:.6f} /km")

        print("\n" + "="*80)
        print("✓ 故障率强度模型建模完成！")
        print("="*80 + "\n")

        # 6. 车辆档案整合 (Vehicle Profiling)
        print("\n" + "="*80)
        print("步骤 6: 车辆档案整合")
        print("="*80)

        # 6.1 获取被 LLM 处理过的车辆 VIN 列表
        print("\n【建立车辆档案】")
        llm_vins = reliability_model.vehicle_profiles['VIN'].tolist()
        print(f"  从 reliability_model 中提取 {len(llm_vins)} 辆被 LLM 处理过的车辆")

        # 6.2 准备车辆数据（用于行为模型）
        df_base_copy = df_base.copy()
        df_base_copy['SETTLE_DATE'] = pd.to_datetime(df_base_copy['SETTLE_DATE'])

        # 识别保养
        if df_llm is None:
            maintenance_keywords = ['保养', '更换机油', '机滤', '三滤', '润滑油']
            df_base_copy['is_maintenance'] = df_base_copy['FAULT_DESC'].str.contains(
                '|'.join(maintenance_keywords), na=False
            )
        else:
            df_base_copy = df_base_copy.merge(
                df_llm[['ID', 'Event_Type']],
                on='ID',
                how='left'
            )
            df_base_copy['is_maintenance'] = (df_base_copy['Event_Type'] == '保养').fillna(False)

        # 聚合
        vehicle_data = df_base_copy.groupby('VIN').agg({
            'REPAIR_MILEAGE': ['min', 'max'],
            'SETTLE_DATE': ['min', 'max'],
            'is_maintenance': 'sum'
        }).reset_index()
        vehicle_data.columns = ['VIN', 'min_mileage', 'max_mileage', 'first_date', 'last_date', 'maint_count']
        vehicle_data['span_days'] = (vehicle_data['last_date'] - vehicle_data['first_date']).dt.days
        vehicle_data.loc[vehicle_data['span_days'] < 30, 'span_days'] = 30

        # 6.3 遍历这些 VIN，构建完整的车辆档案
        print(f"  正在整合车辆档案...")
        profiles_list = []

        for vin in llm_vins:
            # 获取里程和天数
            veh_row = vehicle_data[vehicle_data['VIN'] == vin]
            if veh_row.empty:
                continue

            min_mileage = veh_row.iloc[0]['min_mileage']
            mileage = veh_row.iloc[0]['max_mileage']
            days = veh_row.iloc[0]['span_days']
            maint_count = int(veh_row.iloc[0]['maint_count'])

            # 获取生命周期得分
            weibull_score = model.predict_score(mileage)

            # 获取行为得分（使用修正版：传入 min_mileage）
            usage_score, maint_score = behavior_model.predict_scores(mileage, days, maint_count, min_mileage=min_mileage)

            # 获取可靠性得分
            reliability_score = reliability_model.predict_score(vin)

            # 获取 LLM 记录数
            llm_profile = reliability_model.get_vehicle_profile(vin)
            llm_records = llm_profile['record_count'] if llm_profile else 0

            # 构建档案字典
            profile = {
                'VIN': vin[:10] + '...',  # 只显示前 10 位
                'Weibull_Score': round(weibull_score, 2),
                'Usage_Score': round(usage_score, 2),
                'Maint_Score': round(maint_score, 2),
                'Reliability_Score': round(reliability_score, 2) if reliability_score is not None else None,
                'LLM_Records': llm_records
            }

            profiles_list.append(profile)

        # 6.4 创建最终车辆画像表
        final_vehicle_profiles = pd.DataFrame(profiles_list)

        # 6.5 展示最终车辆画像表
        print(f"\n【最终车辆画像表】")
        print(f"  共 {len(final_vehicle_profiles)} 辆车的完整档案")
        print("\n" + "="*80)
        print(final_vehicle_profiles.to_string(index=False))
        print("="*80 + "\n")

        # 6.6 统计摘要
        print("【档案统计摘要】")
        print(f"  生命周期得分 - 均值: {final_vehicle_profiles['Weibull_Score'].mean():.2f}")
        print(f"  使用强度得分 - 均值: {final_vehicle_profiles['Usage_Score'].mean():.2f} (越低越激烈)")
        print(f"  保养规范度得分 - 均值: {final_vehicle_profiles['Maint_Score'].mean():.2f} (越高越规范)")
        print(f"  可靠性得分 - 均值: {final_vehicle_profiles['Reliability_Score'].mean():.2f} (越高越可靠)")
        print(f"  平均 LLM 记录数: {final_vehicle_profiles['LLM_Records'].mean():.2f}")

        print("\n" + "="*80)
        print("✓ 车辆档案整合完成！")
        print("="*80 + "\n")

        # 7. PCA 组合赋权模型 (第 4 章)
        print("\n" + "="*80)
        print("步骤 7: PCA 组合赋权模型")
        print("="*80)

        # 7.1 拟合 PCA 权重模型
        print("\n【权重计算】")
        weighting_model = PCAWeightingModel()
        weighting_model.fit(final_vehicle_profiles)

        # 7.2 计算最终得分
        print("\n【综合得分计算】")
        final_profiles_with_score = weighting_model.calculate_score(final_vehicle_profiles)

        # 7.3 展示最终得分表 (前 10 名)
        print("\n【最终车辆画像表 - 前 10 名】")
        top_10 = final_profiles_with_score.nlargest(10, 'Final_Score')
        print("\n" + "="*100)
        print(top_10.to_string(index=False))
        print("="*100 + "\n")

        # 7.4 展示车况最好和最差的车
        best_vehicle = final_profiles_with_score.loc[final_profiles_with_score['Final_Score'].idxmax()]
        worst_vehicle = final_profiles_with_score.loc[final_profiles_with_score['Final_Score'].idxmin()]

        print("【车况分析】")
        print("\n🏆 车况最好的车:")
        print(f"  VIN: {best_vehicle['VIN']}")
        print(f"  综合得分: {best_vehicle['Final_Score']:.2f}")
        print(f"  生命周期: {best_vehicle['Weibull_Score']:.2f}")
        print(f"  使用强度 (反转后): {100-best_vehicle['Usage_Score']:.2f} (原始: {best_vehicle['Usage_Score']:.2f})")
        print(f"  保养规范度: {best_vehicle['Maint_Score']:.2f}")
        print(f"  可靠性: {best_vehicle['Reliability_Score']:.2f}")
        print(f"  LLM 记录数: {int(best_vehicle['LLM_Records'])}")

        print("\n⚠ 车况最差的车:")
        print(f"  VIN: {worst_vehicle['VIN']}")
        print(f"  综合得分: {worst_vehicle['Final_Score']:.2f}")
        print(f"  生命周期: {worst_vehicle['Weibull_Score']:.2f}")
        print(f"  使用强度 (反转后): {100-worst_vehicle['Usage_Score']:.2f} (原始: {worst_vehicle['Usage_Score']:.2f})")
        print(f"  保养规范度: {worst_vehicle['Maint_Score']:.2f}")
        print(f"  可靠性: {worst_vehicle['Reliability_Score']:.2f}")
        print(f"  LLM 记录数: {int(worst_vehicle['LLM_Records'])}")

        # 7.5 统计摘要
        print("\n【综合得分统计】")
        print(f"  平均分: {final_profiles_with_score['Final_Score'].mean():.2f}")
        print(f"  中位数: {final_profiles_with_score['Final_Score'].median():.2f}")
        print(f"  最高分: {final_profiles_with_score['Final_Score'].max():.2f}")
        print(f"  最低分: {final_profiles_with_score['Final_Score'].min():.2f}")
        print(f"  标准差: {final_profiles_with_score['Final_Score'].std():.2f}")

        print("\n" + "="*80)
        print("✓ PCA 组合赋权模型完成！")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
