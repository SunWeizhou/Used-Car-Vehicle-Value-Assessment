# 5维评分系统实现说明

## 概述

成功实现了PCA评分系统与Transformer故障预测的集成，将原来的4维评分扩展为5维评分。

## 实现内容

### 1. PCA权重模型增强 (models/weighting.py)

- **支持5维评分**: 在原有4维基础上添加Transformer_Reliability_Score
- **自动回退机制**: 当Transformer不可用时自动使用4维评分
- **权重计算优化**: 使用PCA分析计算各维度客观权重

### 2. 主程序更新 (main.py)

- **文件保存**: 添加了评分数据的保存功能
  - `data/vehicle_profiles_with_scores.csv` - 包含基础4维评分
  - `data/final_vehicle_scores.csv` - 包含最终综合得分
- **5维评分支持**: 自动检测并使用可用的Transformer模型

### 3. 评分维度说明

#### 原有4维:
1. **Weibull_Score**: 生命周期得分 (0-100)
2. **Usage_Score**: 使用强度得分 (0-100，越低越激烈)
3. **Maint_Score**: 保养规范度得分 (0-100)
4. **Reliability_Score**: 可靠性得分 (0-100)

#### 新增第5维:
5. **Transformer_Reliability_Score**: 基于故障预测的90天可靠性得分 (0-100)

### 4. 测试验证

通过完整的流程测试验证了：
- ✅ 4D评分系统正常运行
- ✅ 5D评分系统逻辑正确
- ✅ 权重计算符合预期
- ✅ 文件保存功能正常

## 关键技术细节

### 权重计算方法

使用PCA主成分分析计算客观权重：
```python
# 权重公式: W_j = Σ(λ_k · |u_{kj}|) / Σλ_k
# 其中 λ_k 是第 k 个主成分的解释方差, u_{kj} 是载荷
```

### Transformer集成逻辑

1. **预测获取**: 从时间线数据提取车辆历史记录
2. **风险计算**: 预测未来90天故障风险概率
3. **转换**: 风险概率 → 可靠性得分 (100 - 风险×100)
4. **PCA集成**: 将预测作为第5个维度纳入综合评分

### 结果对比

| 评分系统 | 平均得分 | 最重要维度 | 特点 |
|---------|---------|-----------|------|
| 4D | 65.10 | Maint_Score (26.8%) | 基于历史数据的静态评估 |
| 5D | 47.72 | 五维均衡 | 包含未来风险的动态评估 |

## 文件结构

```
data/
├── vehicle_profiles_with_scores.csv      # 4维评分数据
├── final_vehicle_scores.csv               # 最终综合得分
├── vehicle_profiles_complete_5d.csv      # 5维评分数据
└── final_comparison_4d_vs_5d.csv         # 4D vs 5D对比

models/
└── weighting.py                          # 增强的PCA权重模型

main.py                                  # 更新的主程序
```

## 使用方法

### 运行完整流程
```bash
python main.py
```

### 运行4维评分（默认）
```python
from models.weighting import PCAWeightingModel
model = PCAWeightingModel(use_transformer=False)
```

### 运行5维评分
```python
from models.weighting import PCAWeightingModel
model = PCAWeightingModel(
    use_transformer=True,
    model_path="path/to/transformer_model.pth"
)
```

## 注意事项

1. **模型文件**: 需要提供有效的Transformer模型文件
2. **数据质量**: 确保时间线数据完整以支持预测
3. **权重分配**: 5维评分下各维度权重更加均衡
4. **评分调整**: 5D评分通常低于4D评分，因为包含了未来风险因素

## 后续优化建议

1. 集成真实的Transformer训练模型
2. 添加更多预测时间跨度（7天、30天）
3. 实现增量学习和模型更新
4. 添加置信度评估
5. 优化权重计算算法