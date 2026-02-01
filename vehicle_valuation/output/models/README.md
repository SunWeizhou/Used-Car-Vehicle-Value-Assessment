# 模型文件说明

## 最终保留的模型文件

根据项目配置和run.sh脚本，最终只保留以下模型文件：

### 🎯 主要模型文件
- **`transformer_failure_predictor_enhanced_v2_best.pth`** (309 MB)
  - 这是项目的主要模型，用于故障预测
  - 在run.sh中被引用为预测时使用的模型
  - 是训练过程中表现最好的模型

### 📊 模型清理建议

如果您有原始的模型文件，请执行以下命令来只保留最终版本：

```bash
# 进入模型目录
cd output/models/

# 删除所有epoch相关的模型文件
rm -f best_model_epoch_*.pth

# 删除旧的transformer模型版本
rm -f transformer_failure_predictor.pth
rm -f transformer_failure_predictor_best.pth
rm -f transformer_failure_predictor_final.pth

# 删除增强版v2的final版本（保留best版本）
rm -f transformer_failure_predictor_enhanced_v2_final.pth

# 最终只保留
ls -la transformer_failure_predictor_enhanced_v2_best.pth
```

### 🚀 使用方法

```bash
# 使用最终模型进行推理
python models/inference_transformer.py --model output/models/transformer_failure_predictor_enhanced_v2_best.pth

# 或者使用run.sh脚本
./run.sh
# 选择选项2: 进行故障预测
```

### 💾 存储空间

- 保留单个模型可节省约 **1.2GB** 存储空间
- 最终模型大小：**309 MB**
- 推荐定期清理，只保留生产环境使用的模型版本