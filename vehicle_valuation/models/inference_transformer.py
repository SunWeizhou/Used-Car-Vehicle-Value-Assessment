"""
Transformer故障预测模型推理脚本

该脚本用于使用训练好的Transformer模型进行故障预测。

主要功能：
1. 加载训练好的模型
2. 单车辆故障风险预测
3. 批量预测
4. 结果可视化
5. 维护建议生成

使用方法：
python inference_transformer.py --model_path models/transformer_failure_predictor.pth --data_path data/vehicle_timeline.csv

作者：Claude AI
创建时间：2026-02-01
"""

import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from data_loader import VehicleDataLoader
from models.transformer_predictor import FailurePredictorService

def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """创建输出目录"""
    dirs = {
        'predictions': os.path.join(base_dir, 'predictions'),
        'plots': os.path.join(base_dir, 'plots'),
        'logs': os.path.join(base_dir, 'logs')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def validate_input_data(df: pd.DataFrame) -> bool:
    """验证输入数据"""
    required_columns = ['VIN', 'SETTLE_DATE', 'REPAIR_MILEAGE', 'System', 'Severity']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误：缺少必需的列：{missing_columns}")
        return False

    # 检查数据量
    if len(df) == 0:
        print("错误：输入数据为空")
        return False

    # 检查VIN数据
    if df['VIN'].nunique() == 0:
        print("错误：没有有效的VIN数据")
        return False

    return True

def predict_single_vehicle(
    predictor: FailurePredictorService,
    vin: str,
    df: pd.DataFrame
) -> Dict:
    """预测单辆车的故障风险"""
    # 获取该VIN的历史数据
    vin_data = df[df['VIN'] == vin]

    if len(vin_data) == 0:
        return {
            'vin': vin,
            'error': '未找到该VIN的历史数据'
        }

    try:
        # 预测故障风险
        risk_predictions = predictor.predict_failure_risk(vin_data)

        # 生成维护建议
        recommendations = predictor.generate_maintenance_recommendations(
            risk_predictions, vin_data
        )

        # 计算基础统计信息
        last_mileage = vin_data['REPAIR_MILEAGE'].iloc[-1]
        first_mileage = vin_data['REPAIR_MILEAGE'].iloc[0]
        total_records = len(vin_data)

        # 故障严重程度统计
        severity_counts = vin_data['Severity'].value_counts().to_dict()

        # 系统故障统计
        system_counts = vin_data['System'].value_counts().to_dict()

        result = {
            'vin': vin,
            'risk_predictions': risk_predictions,
            'recommendations': recommendations,
            'statistics': {
                'last_mileage': last_mileage,
                'first_mileage': first_mileage,
                'total_mileage': last_mileage - first_mileage,
                'total_records': total_records,
                'avg_records_per_1000km': total_records / ((last_mileage - first_mileage) / 1000) if last_mileage > first_mileage else 0,
                'days_span': (vin_data['SETTLE_DATE'].max() - vin_data['SETTLE_DATE'].min()).days if len(vin_data) > 1 else 0
            },
            'severity_distribution': severity_counts,
            'system_distribution': system_counts,
            'analysis_timestamp': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        return {
            'vin': vin,
            'error': f'预测失败：{str(e)}'
        }

def predict_batch_vehicles(
    predictor: FailurePredictorService,
    vins: List[str],
    df: pd.DataFrame
) -> List[Dict]:
    """批量预测多辆车的故障风险"""
    results = []

    print(f"开始批量预测 {len(vins)} 辆车...")

    for i, vin in enumerate(vins):
        print(f"正在处理 {i+1}/{len(vins)}: {vin}")

        result = predict_single_vehicle(predictor, vin, df)
        results.append(result)

        # 避免API调用过快
        if i % 10 == 9:
            print("休息1秒...")
            import time
            time.sleep(1)

    return results

def save_predictions(results: List[Dict], output_dir: str):
    """保存预测结果"""
    # 转换为DataFrame
    records = []
    for result in results:
        if 'error' not in result:
            row = {
                'VIN': result['vin'],
                'risk_7d': result['risk_predictions'].get(7, 0),
                'risk_30d': result['risk_predictions'].get(30, 0),
                'risk_90d': result['risk_predictions'].get(90, 0),
                'last_mileage': result['statistics']['last_mileage'],
                'total_records': result['statistics']['total_records'],
                'avg_records_per_1000km': result['statistics']['avg_records_per_1000km'],
                'severity_L3': result['severity_distribution'].get('L3', 0),
                'severity_L2': result['severity_distribution'].get('L2', 0),
                'severity_L1': result['severity_distribution'].get('L1', 0),
                'severity_L0': result['severity_distribution'].get('L0', 0),
                'analysis_timestamp': result['analysis_timestamp']
            }
            records.append(row)

    if records:
        df_predictions = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df_predictions.to_csv(csv_path, index=False)
        print(f"预测结果已保存至：{csv_path}")

        # 保存详细结果
        detailed_path = os.path.join(output_dir, 'detailed_predictions.json')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"详细预测结果已保存至：{detailed_path}")

        return df_predictions
    else:
        print("没有有效的预测结果")
        return None

def visualize_results(results: List[Dict], output_dir: str):
    """可视化预测结果"""
    # 准备数据
    records = []
    for result in results:
        if 'error' not in result:
            row = {
                'VIN': result['vin'],
                'risk_7d': result['risk_predictions'].get(7, 0),
                'risk_30d': result['risk_predictions'].get(30, 0),
                'risk_90d': result['risk_predictions'].get(90, 0)
            }
            records.append(row)

    if not records:
        print("没有数据用于可视化")
        return

    df_plot = pd.DataFrame(records)

    # 创建图表
    plt.style.use('seaborn-v0_8')

    # 1. 风险分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 风险分布直方图
    for i, horizon in enumerate(['7d', '30d', '90d']):
        ax = axes[i//2, i%2]
        sns.histplot(df_plot[f'risk_{horizon}'], bins=30, ax=ax)
        ax.set_title(f'{horizon.replace("d", "天")}故障风险分布')
        ax.set_xlabel('故障概率')
        ax.set_ylabel('车辆数量')

    # 2. 风险相关性热力图
    ax = axes[1, 1]
    correlation_matrix = df_plot[['risk_7d', 'risk_30d', 'risk_90d']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('不同时间跨度风险相关性')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'risk_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"风险分布图已保存至：{plot_path}")

    # 3. 风险等级饼图
    risk_levels = {
        '低风险 (< 10%)': (df_plot['risk_30d'] < 0.1).sum(),
        '中等风险 (10%-30%)': ((df_plot['risk_30d'] >= 0.1) & (df_plot['risk_30d'] < 0.3)).sum(),
        '高风险 (30%-60%)': ((df_plot['risk_30d'] >= 0.3) & (df_plot['risk_30d'] < 0.6)).sum(),
        '极高风险 (> 60%)': (df_plot['risk_30d'] >= 0.6).sum()
    }

    plt.figure(figsize=(10, 8))
    plt.pie(risk_levels.values(), labels=risk_levels.keys(), autopct='%1.1f%%')
    plt.title('30天故障风险等级分布')
    plt.axis('equal')

    pie_path = os.path.join(output_dir, 'risk_levels_pie.png')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"风险等级饼图已保存至：{pie_path}")

def generate_summary_report(results: List[Dict], output_dir: str):
    """生成汇总报告"""
    # 统计信息
    total_vehicles = len(results)
    successful_predictions = len([r for r in results if 'error' not in r])
    failed_predictions = total_vehicles - successful_predictions

    # 风险统计
    risks = []
    for result in results:
        if 'error' not in result:
            risks.extend([
                (7, result['risk_predictions'].get(7, 0)),
                (30, result['risk_predictions'].get(30, 0)),
                (90, result['risk_predictions'].get(90, 0))
            ])

    if risks:
        risk_df = pd.DataFrame(risks, columns=['horizon', 'risk'])
        risk_stats = risk_df.groupby('horizon')['risk'].agg(['mean', 'std', 'min', 'max'])
    else:
        risk_stats = None
        risk_df = None

    # 生成报告
    report = f"""
# Transformer故障预测模型 - 预测报告

## 基本信息
- 预测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总车辆数：{total_vehicles}
- 成功预测：{successful_predictions}
- 预测失败：{failed_predictions}
- 成功率：{successful_predictions/total_vehicles*100:.1f} if total_vehicles > 0 else "N/A"

## 风险预测统计
"""

    if risk_stats is not None:
        report += """
| 时间跨度 | 平均风险 | 标准差 | 最小风险 | 最大风险 |
|----------|----------|--------|----------|----------|
"""
        for horizon, row in risk_stats.iterrows():
            report += f"| {horizon}天 | {row['mean']:.1%} | {row['std']:.1%} | {row['min']:.1%} | {row['max']:.1%} |\n"

    report += """
## 风险等级分布
"""

    if risk_df is not None:
        low_risk = (risk_df['risk'] < 0.1).sum()
        medium_risk = ((risk_df['risk'] >= 0.1) & (risk_df['risk'] < 0.3)).sum()
        high_risk = ((risk_df['risk'] >= 0.3) & (risk_df['risk'] < 0.6)).sum()
        critical_risk = (risk_df['risk'] >= 0.6).sum()

        report += f"""
- 低风险 (< 10%)：{low_risk} 辆车 ({low_risk/len(risk_df)*100:.1f}%)
- 中等风险 (10%-30%)：{medium_risk} 辆车 ({medium_risk/len(risk_df)*100:.1f}%)
- 高风险 (30%-60%)：{high_risk} 辆车 ({high_risk/len(risk_df)*100:.1f}%)
- 极高风险 (> 60%)：{critical_risk} 辆车 ({critical_risk/len(risk_df)*100:.1f}%)
"""

    report += """
## 预测失败原因
"""

    if failed_predictions > 0:
        error_types = {}
        for result in results:
            if 'error' in result:
                error_type = result['error']
                error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            report += f"- {error_type}：{count} 辆车\n"
    else:
        report += "- 无预测失败\n"

    # 保存报告
    report_path = os.path.join(output_dir, 'prediction_summary.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"汇总报告已保存至：{report_path}")
    print(report)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Transformer故障预测模型推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='输入数据路径')
    parser.add_argument('--vins', type=str, nargs='+',
                       help='指定要预测的VIN列表（如果不指定，预测所有VIN）')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='config/inference_config.json',
                       help='推理配置文件路径')

    args = parser.parse_args()

    # 设置日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logging(log_dir)

    # 创建输出目录
    output_dirs = create_output_dirs(args.output_dir)

    logger.info("开始Transformer故障预测模型推理")
    logger.info(f"模型路径：{args.model_path}")
    logger.info(f"数据路径：{args.data_path}")
    logger.info(f"输出目录：{args.output_dir}")

    # 1. 加载数据
    logger.info("加载输入数据...")
    df = pd.read_csv(args.data_path)

    if not validate_input_data(df):
        logger.error("输入数据验证失败")
        return

    logger.info(f"成功加载数据：{len(df)}条记录，{df['VIN'].nunique()}个VIN")

    # 2. 初始化预测器
    logger.info("初始化预测器...")
    try:
        predictor = FailurePredictorService(args.model_path)
        model_summary = predictor.get_model_summary()
        logger.info(f"模型加载成功")
        logger.info(f"- 总参数量：{model_summary['total_parameters']:,}")
        logger.info(f"- 模型大小：{model_summary['model_size_mb']:.1f} MB")
        logger.info(f"- 预测设备：{model_summary['device']}")
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return

    # 3. 确定预测的VIN列表
    if args.vins:
        logger.info(f"输入的VIN列表: {args.vins}")
        logger.info(f"数据中的VIN数量: {df['VIN'].nunique()}")
        logger.info(f"前5个VIN: {df['VIN'].unique()[:5].tolist()}")
        vins_to_predict = [vin for vin in args.vins if vin in df['VIN'].values]
        logger.info(f"指定预测 {len(vins_to_predict)} 个VIN")
    else:
        vins_to_predict = df['VIN'].unique().tolist()
        logger.info(f"预测所有 {len(vins_to_predict)} 个VIN")

    # 4. 执行预测
    results = predict_batch_vehicles(predictor, vins_to_predict, df)

    # 5. 保存结果
    logger.info("保存预测结果...")
    df_predictions = save_predictions(results, output_dirs['predictions'])

    # 6. 可视化
    if df_predictions is not None:
        logger.info("生成可视化图表...")
        visualize_results(results, output_dirs['plots'])

    # 7. 生成汇总报告
    logger.info("生成汇总报告...")
    generate_summary_report(results, output_dirs['predictions'])

    logger.info("推理完成！")

if __name__ == '__main__':
    main()