#!/bin/bash

# 车辆故障预测系统启动脚本
# Vehicle Fault Prediction System Launcher

echo "🚗 车辆故障预测系统启动脚本"
echo "Vehicle Fault Prediction System Launcher"
echo "========================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖包..."
python3 -c "import torch, pandas, numpy, sklearn; print('✅ 所有必要依赖已安装')" 2>/dev/null || {
    echo "❌ 缺少依赖包，请运行：pip install torch pandas numpy scikit-learn"
    exit 1
}

# 检查数据文件
if [ ! -f "data/vehicle_timeline.csv" ]; then
    echo "⚠️  未找到数据文件 data/vehicle_timeline.csv"
    echo "请确保数据文件存在"
fi

# 显示可用选项
echo ""
echo "🎯 请选择要执行的操作："
echo "1) 训练增强版模型 (推荐)"
echo "2) 进行故障预测"
echo "3) 运行完整系统"
echo "4) 查看项目文档"
echo "5) 退出"

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "🚀 开始训练增强版模型..."
        python3 train_enhanced_v2.py --config config/train_enhanced_v2_config.json
        ;;
    2)
        echo "🔍 进行故障预测..."
        if [ -f "output/models/transformer_failure_predictor_enhanced_v2_best.pth" ]; then
            python3 inference_transformer.py --model output/models/transformer_failure_predictor_enhanced_v2_best.pth
        else
            echo "❌ 未找到训练好的模型，请先运行训练"
        fi
        ;;
    3)
        echo "📊 运行完整系统..."
        python3 main.py
        ;;
    4)
        echo "📖 查看项目文档..."
        if command -v bat &> /dev/null; then
            bat README.md CLAUDE.md
        elif command -v less &> /dev/null; then
            less README.md
        else
            cat README.md
        fi
        ;;
    5)
        echo "👋 退出程序"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 操作完成！"