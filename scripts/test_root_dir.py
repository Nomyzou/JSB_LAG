#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.JSBSim.utils.utils import get_root_dir

if __name__ == "__main__":
    root_dir = get_root_dir()
    print(f"Root directory: {root_dir}")
    
    # 检查模型文件是否存在
    model_path = root_dir + '/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1_selfplay_from_baseline/run3/actor_latest.pt'
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    # 检查当前工作目录
    print(f"当前工作目录: {os.getcwd()}") 