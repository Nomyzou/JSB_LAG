# 2v1 空战渲染脚本使用说明

## 概述
这些脚本用于渲染训练好的2对1空战MAPPO模型，生成ACMI文件用于可视化分析。

## 可用的渲染脚本

### 1. render_2v1_400steps.sh (推荐)
专门用于渲染400步的2对1空战场景。

```bash
./scripts/render_2v1_400steps.sh
```

### 2. render_2v1_mappo.sh
通用2v1渲染脚本，默认渲染400步。

```bash
./scripts/render_2v1_mappo.sh
```

## 手动运行渲染

如果需要自定义参数，可以直接运行Python脚本：

```bash
python renders/render_2v1_mappo.py \
    --model-dir scripts/results/MAPPOTraining2v1/2v1/NoWeapon/MAPPOTraining/mappo/v1_2v1_mappo_training/wandb/latest-run/files \
    --output custom_output.txt.acmi \
    --scenario 2v1/NoWeapon/MAPPOTraining \
    --target-steps 400 \
    --use-latest
```

## 参数说明

- `--model-dir`: 训练好的模型文件目录
- `--output`: 输出的ACMI文件名
- `--scenario`: 场景名称
- `--target-steps`: 目标渲染步数（默认400）
- `--use-latest`: 使用最新的模型文件
- `--episode`: 指定特定的episode模型（可选）

## 输出文件

渲染完成后会生成：
1. ACMI文件：用于Tacview等工具可视化
2. 控制台日志：显示渲染进度和结果

## 注意事项

1. 确保模型文件存在且可访问
2. 渲染会持续到指定的步数，即使episode提前结束
3. 生成的ACMI文件包含静态对象标记，便于分析战场态势
4. 建议使用GPU进行渲染以提高性能

## 故障排除

如果遇到问题：
1. 检查模型文件路径是否正确
2. 确认环境依赖已安装
3. 查看控制台错误信息
4. 检查CUDA设备是否可用 