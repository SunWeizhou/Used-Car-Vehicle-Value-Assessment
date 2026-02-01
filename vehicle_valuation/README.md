# 🚗 二手车残值评估系统

基于多维度健康 profiling 的车辆残值评估系统，结合 LLM 维修记录分类、统计建模和可靠性工程原理，评估车辆状况和价值。

## 📁 项目结构

```
vehicle_valuation/
├── 🎯 main.py                      # 主程序入口
├── 🤖 models/main_with_transformer.py # Transformer模型版本
├── 📊 data/                        # 输入数据和LLM处理结果
│   ├── 上汽跃进_燃油_baseinfo.csv   # 基础维修记录
│   ├── 上汽跃进_燃油_parts_info.csv # 配件更换记录
│   ├── 上汽跃进_燃油_time_info.csv  # 工时记录
│   ├── llm_parsed_results.csv       # LLM分类结果
│   ├── vehicle_profiles.csv        # 车辆级统计数据
│   └── vehicle_timeline.csv        # 时间线数据
├── 🧠 models/                      # 核心模型和ML组件
│   ├── lifecycle.py               # Weibull生命周期模型
│   ├── behavior.py                # 使用行为评分模型
│   ├── reliability.py             # 可靠性故障率模型
│   ├── weighting.py               # PCA权重模型
│   ├── transformer_predictor.py   # Transformer故障预测
│   ├── inference_transformer.py   # 推理工具
│   ├── data_augmentation.py       # 数据增强
│   └── data_loader.py            # 数据加载器
├── 🛠️ utils/                       # 工具函数
│   ├── preprocessing.py           # 数据预处理
│   ├── llm_structuring.py        # LLM维修记录分类
│   └── math_tools.py             # 数学工具
├── 📦 output/                      # 输出和模型
│   ├── models/                   # 训练好的模型
│   └── predictions/              # 预测结果
├── ⚙️ config/                      # 配置文件
├── 📝 logs/                        # 日志文件
├── ▶️ run.sh                       # 运行脚本
├── 📖 CLAUDE.md                   # 详细文档
├── 📋 README.md                   # 本说明文件
└── 🔑 .env                        # 环境变量
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch pandas numpy scikit-learn matplotlib

# 设置API密钥（如果使用LLM）
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 2. 运行完整流程

```bash
# 运行主要估值系统（推荐）
python main.py

# 或使用Transformer模型版本
python models/main_with_transformer.py
```

### 3. LLM数据预处理

```bash
# 处理维修记录数据（首次运行或新增数据时）
python utils/llm_structuring.py
```

## 📊 模型性能

- **最佳验证损失**: 0.446055
- **模型参数量**: 2,613万
- **训练序列数**: 3,206个（包含数据增强）
- **支持预测时间跨度**: 7天、30天、90天
- **性能等级**: 工业应用可用水平

## 🎯 主要功能

### 四维健康评分系统
1. **生命周期评分** (Lifestyle): 基于Weibull生存分析的车辆预期寿命评估
2. **使用行为评分** (Behavior): 基于ECDF的使用强度和维护规律性评估
3. **维护质量评分** (Maintenance): 基于维修记录密度的维护质量评估
4. **可靠性评分** (Reliability): 基于故障率强度的可靠性评估

### 其他功能
5. **LLM维修分类**: 使用深度求索大模型自动分类维修记录严重程度
6. **PCA加权融合**: 客观权重计算，综合四个维度得出最终估值
7. **故障概率预测**: 基于Transformer的未来故障概率预测

## 🔧 配置说明

### 模型配置
- `d_model`: 512 (隐藏层维度)
- `num_layers`: 8 (Transformer层数)
- `nhead`: 8 (注意力头数)
- `dropout`: 0.15 ( dropout率)

### 训练配置
- `batch_size`: 32
- `learning_rate`: 0.00005
- `epochs`: 100 (早停机制)
- `patience`: 15

### 数据配置
- `window_size`: 6 (滑动窗口大小)
- `stride`: 3 (步长)
- `prediction_horizons`: [7, 30, 90] (预测时间跨度)

## 📈 输出文件

### 模型文件
- `transformer_failure_predictor_enhanced_v2_best.pth` - 最佳模型（309 MB，主要使用）

### 训练记录
- `training_history.json` - 训练历史和配置
- `training_curves_enhanced.png` - 训练曲线图

## 🛠️ 开发指南

### 添加新特征
1. 在 `data_loader.py` 中添加特征提取逻辑
2. 更新 `models/transformer_predictor.py` 中的特征维度
3. 调整配置文件中的相关参数

### 修改模型结构
1. 编辑 `models/transformer_predictor.py`
2. 更新模型参数（d_model, num_layers等）
3. 重新训练模型

### 数据增强
1. 调整 `data_augmentation.py` 中的增强策略
2. 更新配置文件中的增强参数
3. 重新运行训练

## ⚠️ 注意事项

1. **模型性能**: 当前模型达到工业应用可用水平，建议在生产环境中配合监控使用
2. **数据质量**: 确保输入数据的完整性和准确性
3. **定期更新**: 建议定期收集新数据并重新训练模型
4. **资源消耗**: 模型文件较大(309MB)，请确保足够的存储空间
5. **模型清理**: 为节省存储空间，建议只保留最终上线的模型文件

## 📞 支持

如有问题，请查看：
1. `CLAUDE.md` - 详细的项目文档
2. `scripts/` - 相关测试和调试脚本
3. `logs/` - 运行日志文件

## 📄 许可证

本项目仅供学习和研究使用。