import pandas as pd
import numpy as np
from math import radians, cos, sin, tan, degrees, pi, exp

def body_to_earth_rotation_matrix(roll, pitch, yaw):
    """
    计算从机体坐标系到地球坐标系(东北天)的旋转矩阵
    
    在标准航空定义中:
    - 机体坐标系: x轴沿机头向前，y轴沿右翼，z轴向下
    - 大地坐标系: x轴指向东方，y轴指向北方，z轴指向天空
    
    参数:
        roll: 横滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
    
    返回:
        3x3 旋转矩阵
    """
    # 分别计算旋转矩阵的元素（基于ZYX顺序的欧拉角转换）
    sin_roll, cos_roll = sin(roll), cos(roll)
    sin_pitch, cos_pitch = sin(pitch), cos(pitch)
    sin_yaw, cos_yaw = sin(yaw), cos(yaw)
    
    # 构建旋转矩阵（从机体坐标系到地球坐标系）
    # 注意: 我们需要将机体坐标系(x:前，y:右，z:下)转换到大地坐标系(x:东，y:北，z:上)
    rot_matrix = np.array([
        [cos_pitch*sin_yaw, cos_roll*cos_yaw + sin_roll*sin_pitch*sin_yaw, sin_roll*cos_yaw - cos_roll*sin_pitch*sin_yaw],
        [cos_pitch*cos_yaw, -cos_roll*sin_yaw + sin_roll*sin_pitch*cos_yaw, -sin_roll*sin_yaw - cos_roll*sin_pitch*cos_yaw],
        [sin_pitch, -sin_roll*cos_pitch, cos_roll*cos_pitch]
    ])
    
    return rot_matrix

def normalize_angle(angle):
    """
    规范化角度到[-pi, pi]范围
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

def parse_flight_log_to_dataframe(file_path):
    """
    解析飞行日志文件为pandas DataFrame。
    假设文件为CSV格式，包含飞行数据。
    """
    # 直接使用pandas读取CSV文件
    df = pd.read_csv(file_path)
    return df

def angle_advantage(phi_u, phi_q):
    """
    角度优势函数 f_ang
    phi_u: 方位角（本方）或进入角（敌方），单位：弧度
    phi_q: 进入角（本方）或方位角（敌方），单位：弧度
    """
    return 1 - (abs(phi_u) + abs(phi_q)) / (2 * np.pi)

def distance_advantage(R, Rw=5.0, sigma=2.0):
    """
    距离优势函数 f_d
    R: 两机距离，单位：km
    Rw: 有效攻击距离，单位：km
    sigma: 分布标准差，单位：km
    """
    if R <= Rw:
        return 1.0
    else:
        return np.exp(-((R - Rw) ** 2) / (2 * sigma ** 2))

def calc_vop(R, vT, vmax=0.3, Rw=5.0):
    """
    计算最佳攻击速度 vop
    R: 两机距离，单位km
    vT: 目标机速度，单位km/s
    vmax: 我方最大速度，单位km/s，固定为0.3
    Rw: 有效攻击距离，单位km
    """
    if R > Rw:
        return vT + (vmax - vT) * (1 - np.exp(-(R - Rw) / Rw))
    else:
        return vT

def speed_advantage(v, v_op):
    """
    速度优势函数 f_v
    v: 当前速度，单位：km/s
    v_op: 最佳攻击速度，单位：km/s
    """
    if v_op == 0:
        return 0.0
    return (v / v_op) * np.exp(-2 * abs(v - v_op) / v_op)

def height_advantage(delta_h, sigma_h=0.5):
    """
    高度优势函数 f_h
    delta_h: 高度差（本方-敌方），单位：km
    sigma_h: 最佳高度差标准差，单位：km
    """
    if delta_h <= 0:
        return np.exp(-(delta_h ** 2) / (2 * sigma_h ** 2))
    elif 0 < delta_h <= sigma_h:
        return 1.0
    else:
        return np.exp(-((delta_h - sigma_h) ** 2) / (2 * sigma_h ** 2))

def calculate_engagement_angles(df):
    """
    计算交战角度：方位角和进入角
    使用标准东北天坐标系:
    - x轴指向东方
    - y轴指向北方
    - z轴指向天空
    
    距离单位转换为千米(km)
    
    参数:
        df: 包含飞行数据的DataFrame
    """
    # 定义原点坐标
    ORIGIN_LON = 120.0  # 经度原点 (度)
    ORIGIN_LAT = 60.0   # 纬度原点 (度)
    ORIGIN_ALT = 0.0    # 高度原点 (米)
    
    # 地球半径（千米）
    R_earth = 6371.0  # 地球平均半径，单位千米
    
    # 创建结果数据框
    result_df = pd.DataFrame()
    
    # 参数
    Rw = 5.0      # 有效攻击距离 km
    sigma = 2.0   # 距离分布标准差 km
    vmax = 0.3    # 我方最大速度 km/s
    sigma_h = 0.5 # 高度差标准差 km
    
    # 提取每个时间步的数据并计算角度
    for step in df['step'].unique():
        # 获取当前时间步的数据
        step_data = df[df['step'] == step].copy()
        
        # 确保我们有两个飞机的数据
        if len(step_data) != 2 or not (0 in step_data['agent_id'].values and 1 in step_data['agent_id'].values):
            continue
            
        # 提取我方飞机(agent_id=0)和敌方飞机(agent_id=1)的数据
        agent0_data = step_data[step_data['agent_id'] == 0].iloc[0]
        agent1_data = step_data[step_data['agent_id'] == 1].iloc[0]
        
        # --- 坐标转换: 将经纬度转换为以原点为中心的直角坐标系(km) ---
        
        # 步骤1: 将经纬度转换为弧度
        lat0_rad = radians(agent0_data['lat'])
        lon0_rad = radians(agent0_data['lon'])
        lat1_rad = radians(agent1_data['lat'])
        lon1_rad = radians(agent1_data['lon'])
        origin_lat_rad = radians(ORIGIN_LAT)
        origin_lon_rad = radians(ORIGIN_LON)
        
        # 步骤2: 计算相对于原点的位置（近似平面投影，适用于小范围）
        # 北向距离(y轴): 纬度差乘地球半径
        y0 = R_earth * (lat0_rad - origin_lat_rad)
        y1 = R_earth * (lat1_rad - origin_lat_rad)
        
        # 东向距离(x轴): 经度差乘地球半径乘cos(纬度)
        x0 = R_earth * (lon0_rad - origin_lon_rad) * cos(origin_lat_rad)
        x1 = R_earth * (lon1_rad - origin_lon_rad) * cos(origin_lat_rad)
        
        # 高度(z轴): 相对于原点的高度，转换为千米(向上为正)
        z0 = (agent0_data['alt'] - ORIGIN_ALT) / 1000.0
        z1 = (agent1_data['alt'] - ORIGIN_ALT) / 1000.0
        
        # --- 计算从我方到敌方的相对位置向量 ---
        rel_pos_vector = np.array([x1 - x0, y1 - y0, z1 - z0])  # 从我方指向敌方的向量
        rel_pos_2d = np.array([rel_pos_vector[0], rel_pos_vector[1]])  # 水平面投影
        
        # --- 速度向量计算 ---
        # 从机体坐标系转换到大地坐标系
        
        # 1. 我方飞机
        roll0_rad = agent0_data['roll']
        pitch0_rad = agent0_data['pitch']
        yaw0_rad = agent0_data['yaw']
        
        # 机体坐标系速度
        # 在JSBSim中: 
        # - vx: 沿机体前向的速度 (指向机头)
        # - vy: 沿机体右翼方向的速度 (指向右翼)
        # - vz: 沿机体下方向的速度 (指向地面)
        body_vel0 = np.array([agent0_data['vx'], agent0_data['vy'], agent0_data['vz']]) / 1000.0  # 转换为km/s
        
        # 计算旋转矩阵并转换到大地坐标系
        rotation_matrix0 = body_to_earth_rotation_matrix(roll0_rad, pitch0_rad, yaw0_rad)
        v0 = rotation_matrix0.dot(body_vel0)  # 转换到东北天坐标系
        v0_2d = np.array([v0[0], v0[1]])  # 水平面投影 (东-北平面)
        
        # 2. 敌方飞机
        roll1_rad = agent1_data['roll']
        pitch1_rad = agent1_data['pitch']
        yaw1_rad = agent1_data['yaw']
        
        # 机体坐标系速度
        body_vel1 = np.array([agent1_data['vx'], agent1_data['vy'], agent1_data['vz']]) / 1000.0  # 转换为km/s
        
        # 计算旋转矩阵并转换到大地坐标系
        rotation_matrix1 = body_to_earth_rotation_matrix(roll1_rad, pitch1_rad, yaw1_rad)
        v1 = rotation_matrix1.dot(body_vel1)  # 转换到东北天坐标系
        v1_2d = np.array([v1[0], v1[1]])  # 水平面投影 (东-北平面)
        
        # --- 计算方位角和进入角 ---
        
        # 计算向量的单位向量形式
        rel_pos_unit = rel_pos_2d / np.linalg.norm(rel_pos_2d)
        v0_unit = v0_2d / np.linalg.norm(v0_2d)
        v1_unit = v1_2d / np.linalg.norm(v1_2d)
        
        # 1. 方位角(fi): 我方速度矢量与双方飞机质点连线夹角
        # 使用点积计算两个单位向量的夹角
        cos_azimuth = np.clip(np.dot(v0_unit, rel_pos_unit), -1.0, 1.0)
        azimuth_rad = np.arccos(cos_azimuth)
        
        # 确定方位角的符号(是否偏左或偏右)
        cross_azimuth = np.cross(v0_unit, rel_pos_unit)
        if cross_azimuth < 0:
            azimuth_rad = -azimuth_rad
            
        # 2. 进入角: 敌方飞机速度矢量与双方飞机质点连线延长线的夹角
        # 敌方到我方的向量 = -rel_pos_vector
        neg_rel_pos_unit = rel_pos_unit
        cos_aspect = np.clip(np.dot(v1_unit, neg_rel_pos_unit), -1.0, 1.0)
        aspect_rad = np.arccos(cos_aspect)
        
        # 确定进入角的符号
        cross_aspect = np.cross(v1_unit, neg_rel_pos_unit)
        if cross_aspect < 0:
            aspect_rad = -aspect_rad
        
        # 将弧度转换为度，范围为[-180, 180]
        azimuth_deg = np.rad2deg(azimuth_rad)
        aspect_angle_deg = np.rad2deg(aspect_rad)
        # --- 敌方角度直接由我方角度推算 ---
        azimuth_deg_enemy = ((180 - aspect_angle_deg + 180) % 360) - 180
        aspect_angle_deg_enemy = ((180 - azimuth_deg + 180) % 360) - 180
        
        # --- 用于调试的额外信息 ---
        # 计算大地坐标系中的航向角(相对于正北方向的夹角，顺时针为正)
        v0_bearing = np.rad2deg(np.arctan2(v0[0], v0[1]))  # 东/北 -> 相对于北的夹角
        v1_bearing = np.rad2deg(np.arctan2(v1[0], v1[1]))
        los_bearing = np.rad2deg(np.arctan2(rel_pos_vector[0], rel_pos_vector[1]))
        
        # 计算两机距离
        distance = np.linalg.norm(rel_pos_vector)
        
        # 计算两机速度大小
        v0_abs = np.linalg.norm(v0_2d)
        v1_abs = np.linalg.norm(v1_2d)
        # 计算高度差
        delta_h_0 = z0 - z1  # 我方-敌方
        delta_h_1 = z1 - z0  # 敌方-我方
        # 角度转弧度
        azimuth_rad = normalize_angle(azimuth_rad)
        aspect_rad = normalize_angle(aspect_rad)
        # --- 优势函数 ---
        # 我方
        f_ang_0 = angle_advantage(azimuth_rad, aspect_rad)
        f_d_0 = distance_advantage(distance, Rw, sigma)
        vop_0 = calc_vop(distance, v1_abs, vmax, Rw)
        f_v_0 = speed_advantage(v0_abs, vop_0)
        f_h_0 = height_advantage(delta_h_0, sigma_h)
        # 敌方（角色互补，不再重复计算角度）
        f_ang_1 = angle_advantage(np.deg2rad(azimuth_deg_enemy), np.deg2rad(aspect_angle_deg_enemy))
        f_d_1 = distance_advantage(distance, Rw, sigma)
        vop_1 = calc_vop(distance, v0_abs, vmax, Rw)
        f_v_1 = speed_advantage(v1_abs, vop_1)
        f_h_1 = height_advantage(delta_h_1, sigma_h)
        # --- 态势判断与总体函数 ---
        situation_0 = judge_battle_situation(azimuth_deg, aspect_angle_deg)
        situation_1 = judge_battle_situation(azimuth_deg_enemy, aspect_angle_deg_enemy)
        overall_0 = calc_overall_situation(f_ang_0, f_d_0, f_v_0, f_h_0, situation_0)
        overall_1 = calc_overall_situation(f_ang_1, f_d_1, f_v_1, f_h_1, situation_1)
        
        # 将结果添加到数据框中
        result_row = {
            'step': step,
            '方位角_fi_deg': azimuth_deg,
            '进入角_deg': aspect_angle_deg,
            '敌方_方位角_deg': azimuth_deg_enemy,
            '敌方_进入角_deg': aspect_angle_deg_enemy,
            '我方_x东_km': x0,
            '我方_y北_km': y0,
            '我方_z天_km': z0,
            '敌方_x东_km': x1,
            '敌方_y北_km': y1,
            '敌方_z天_km': z1,
            '我方_vx东_kmps': v0[0],
            '我方_vy北_kmps': v0[1],
            '我方_vz天_kmps': v0[2],
            '敌方_vx东_kmps': v1[0],
            '敌方_vy北_kmps': v1[1],
            '敌方_vz天_kmps': v1[2],
            '我方_roll_deg': degrees(agent0_data['roll']),
            '我方_pitch_deg': degrees(agent0_data['pitch']),
            '我方_yaw_deg': degrees(agent0_data['yaw']),
            '敌方_roll_deg': degrees(agent1_data['roll']),
            '敌方_pitch_deg': degrees(agent1_data['pitch']),
            '敌方_yaw_deg': degrees(agent1_data['yaw']),
            '我方航向角_deg': v0_bearing,
            '敌方航向角_deg': v1_bearing,
            '视线角_deg': los_bearing,
            '距离_km': distance,
            # 优势函数
            '我方_角度优势': f_ang_0,
            '我方_距离优势': f_d_0,
            '我方_速度优势': f_v_0,
            '我方_高度优势': f_h_0,
            '敌方_角度优势': f_ang_1,
            '敌方_距离优势': f_d_1,
            '敌方_速度优势': f_v_1,
            '敌方_高度优势': f_h_1,
            '我方_态势类型': situation_0,
            '敌方_态势类型': situation_1,
            '我方_总体态势值': overall_0,
            '敌方_总体态势值': overall_1
        }
        result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)

        
    
    return result_df

def judge_battle_situation(azimuth_deg, aspect_angle_deg):
    """
    严格按照表3-3的区间标准进行判定：
    - 我方优势（a）：方位角φU ∈ [0, π/2] 且 进入角qU ∈ [0, π/2]
    - 双方优势（c）：方位角φU ∈ [0, π/2] 且 进入角qU ∈ [π/2, π]
    - 敌方优势（b）：方位角φU ∈ [π/2, π] 且 进入角qU ∈ [0, π/2]
    - 双方劣势（d）：方位角φU ∈ [π/2, π] 且 进入角qU ∈ [π/2, π]
    输入角度为度，需转换为弧度并取绝对值后映射到[0, π]
    """
    # 取绝对值并转为弧度
    az = abs(np.deg2rad(azimuth_deg))
    asp = abs(np.deg2rad(aspect_angle_deg))
    half_pi = np.pi / 2
    pi_ = np.pi
    if 0 <= az <= half_pi and 0 <= asp <= half_pi:
        return 'a'  # 我方优势
    elif 0 <= az <= half_pi and half_pi < asp <= pi_:
        return 'c'  # 双方优势
    elif half_pi < az <= pi_ and 0 <= asp <= half_pi:
        return 'b'  # 敌方优势
    elif half_pi < az <= pi_ and half_pi < asp <= pi_:
        return 'd'  # 双方劣势
    else:
        return 'd'  # 兜底，理论不会到这里

# 动态权重表
DYNAMIC_WEIGHTS = {
    'a': [0.332, 0.291, 0.209, 0.168],
    'b': [0.325, 0.210, 0.278, 0.287],
    'c': [0.239, 0.328, 0.278, 0.155],
    'd': [0.111, 0.313, 0.403, 0.173],
}
# 常权重
STATIC_WEIGHTS = [0.8, 0.1, 0.05, 0.05]

def calc_overall_situation(f_ang, f_d, f_v, f_h, situation):
    """
    计算总体态势函数值
    :param f_ang: 角度优势
    :param f_d: 距离优势
    :param f_v: 速度优势
    :param f_h: 高度优势
    :param situation: 当前态势类型（'a','b','c','d'）
    :return: 总体态势函数值
    """
    dynamic_w = DYNAMIC_WEIGHTS[situation]
    static_w = STATIC_WEIGHTS
    # 动态权重和常权重各占50%
    final_w = [0.5 * dw + 0.5 * sw for dw, sw in zip(dynamic_w, static_w)]
    vals = [f_ang, f_d, f_v, f_h]
    return sum([v * w for v, w in zip(vals, final_w)])

def main():
    """
    主函数：读取数据，计算角度，保存结果
    """
    log_file = 'flight_log.txt'
    output_csv = 'engagement_results5.csv'
    
    # 步骤1: 解析日志文件
    print(f"解析日志文件: {log_file}...")
    flight_data_df = parse_flight_log_to_dataframe(log_file)
    
    # 步骤2: 执行计算
    if not flight_data_df.empty:
        print("计算交战角度及优势函数（使用标准东北天坐标系）...")
        results_df = calculate_engagement_angles(flight_data_df)
        
        # 步骤3: 保存结果到CSV
        print(f"保存结果到 {output_csv}...")
        results_df.to_csv(output_csv, index=False, float_format='%.4f')
        print("计算完成。")
    else:
        print(f"无法从 {log_file} 中提取有效数据。")


if __name__ == '__main__':
    main()