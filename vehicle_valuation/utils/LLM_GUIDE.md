# LLM 结构化处理模块使用指南

## 功能说明

`llm_structuring.py` 模块使用 DeepSeek API 对维修记录进行智能结构化分析，自动提取：
- **Event_Type**: 事件类型（保养/维修）
- **System**: 受损系统（发动机/变速箱/制动/电气等）
- **Severity**: 严重程度（L0-L3）
- **Reasoning**: 判断理由

## 安装依赖

```bash
pip install openai pandas
```

## 获取 API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入 API Keys 页面创建密钥
4. 复制 API Key

## 使用方法

### 方式 1: 在 Python 脚本中使用

```python
from vehicle_valuation.utils.preprocessing import load_and_clean_data
from vehicle_valuation.utils.llm_structuring import process_sample_batch

# 1. 加载数据
df_base, df_parts, df_time = load_and_clean_data("vehicle_valuation/data")

# 2. 设置 API Key
import os
api_key = os.getenv('DEEPSEEK_API_KEY')  # 从环境变量读取
# 或者直接设置（不推荐）
# api_key = "sk-your-api-key-here"

# 3. 处理样本
results_df = process_sample_batch(
    base_df=df_base,
    parts_df=df_parts,
    time_df=df_time,
    api_key=api_key,
    sample_size=200  # 处理 200 条样本
)

# 4. 查看结果
print(results_df.head())
```

### 方式 2: 使用测试脚本

```bash
# 1. 设置环境变量
export DEEPSEEK_API_KEY="sk-your-api-key-here"

# 2. 运行测试脚本
python test_llm.py
```

### 方式 3: 测试 API 连接

```python
from vehicle_valuation.utils.llm_structuring import test_llm_connection

test_llm_connection("sk-your-api-key-here")
```

## 输出格式

处理完成后，结果将保存在 `vehicle_valuation/data/llm_parsed_results.csv`：

| ID | Event_Type | System | Severity | Reasoning |
|----|-----------|--------|----------|-----------|
| xxx | 保养 | 常规耗材 | L0 | 更换机油机滤，常规保养 |
| xxx | 维修 | 制动 | L2 | 更换刹车片，一般维修 |
| xxx | 维修 | 发动机 | L3 | 发动机大修，重大维修 |

## 严重程度说明

- **L0 (保养)**: 常规保养项目（机油、滤芯、打黄油等）
- **L1 (轻微)**: 易损件更换（灯泡、雨刷、保险丝等）
- **L2 (一般)**: 部件更换（刹车片、电瓶、轮胎等）
- **L3 (重大)**: 总成大修或严重故障（发动机、变速箱、事故维修等）

## 系统分类

- 发动机
- 变速箱
- 制动
- 电气
- 车身
- 常规耗材
- 底盘
- 空调
- 其他

## 费用估算

DeepSeek API 定价（仅供参考）：
- 输入: ¥1 / 百万 tokens
- 输出: ¥2 / 百万 tokens

估算：处理 200 条记录约需 ¥0.5-1.0

## 注意事项

1. **API 安全**: 不要将 API Key 提交到 Git 仓库
2. **速率限制**: 注意 API 调用频率限制，建议添加延时
3. **错误处理**: 模块已包含异常处理，失败的记录会标记为 ERROR
4. **进度反馈**: 每处理 10 条记录会打印一次进度

## 扩展功能

如需处理完整数据集，建议：

1. 分批处理（每次 500-1000 条）
2. 添加进度条（tqdm）
3. 保存中间结果
4. 实现断点续传

示例代码：

```python
batch_size = 1000
total_records = len(df_base)

for i in range(0, total_records, batch_size):
    batch_df = df_base.iloc[i:i+batch_size]
    results = process_sample_batch(batch_df, df_parts, df_time, api_key, sample_size=len(batch_df))
    # 保存批次结果
    results.to_csv(f"results_batch_{i}.csv", index=False)
```
