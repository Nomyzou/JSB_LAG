#!/usr/bin/env python
import os

# 这里我们直接硬编码模型路径，而不是使用get_root_dir()函数
base_path = "/Users/zoumaoming/LAG"
model_path = base_path + '/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1_selfplay_from_baseline/run3/actor_latest.pt'
print(f"模型路径: {model_path}")
print(f"模型文件存在: {os.path.exists(model_path)}")

# 也测试没有scripts前缀的路径
alt_model_path = base_path + '/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1_selfplay_from_baseline/run3/actor_latest.pt'
print(f"替代模型路径: {alt_model_path}")
print(f"替代模型文件存在: {os.path.exists(alt_model_path)}")

print(f"当前工作目录: {os.getcwd()}") 