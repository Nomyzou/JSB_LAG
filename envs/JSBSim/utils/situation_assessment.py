import numpy as np
from envs.JSBSim.core.catalog import Property


def get_situation_adv(ego_state, enm_state, weights=(0.4, 0.3, 0.3)):
    """
    Calculate situation advantage value of ego plane over enm plane.
    
    Args:
        ego_state (np.ndarray): state of ego plane, with format of [lat, lon, alt, roll, pitch, yaw, u, v, w, ...]
        enm_state (np.ndarray): state of enm plane, with same format as ego_state
        weights (tuple): weights for angle, altitude, and velocity advantage
    
    Returns:
        float: situation advantage value
    """
    angle_adv = get_angle_adv(ego_state, enm_state)
    alt_adv = get_alt_adv(ego_state, enm_state)
    vel_adv = get_vel_adv(ego_state, enm_state)
    
    adv = weights[0] * angle_adv + weights[1] * alt_adv + weights[2] * vel_adv
    return adv


def get_angle_adv(ego_state, enm_state):
    """
    Calculate angle advantage.
    """
    ego_v = np.array([ego_state[Property.VEL_U.value], ego_state[Property.VEL_V.value], ego_state[Property.VEL_W.value]])
    enm_v = np.array([enm_state[Property.VEL_U.value], enm_state[Property.VEL_V.value], enm_state[Property.VEL_W.value]])
    
    # Vector from ego to enemy
    los_vec = np.array([enm_state[Property.LAT.value] - ego_state[Property.LAT.value],
                        enm_state[Property.LON.value] - ego_state[Property.LON.value],
                        enm_state[Property.ALT.value] - ego_state[Property.ALT.value]])
    los_vec_norm = np.linalg.norm(los_vec)
    if los_vec_norm < 1e-6:
        return 0.0

    los_vec /= los_vec_norm

    # Antenna Train Angle (ATA) for ego
    ego_v_norm = np.linalg.norm(ego_v)
    if ego_v_norm < 1e-6:
        ata = np.pi
    else:
        ata = np.arccos(np.clip(np.dot(ego_v, los_vec) / ego_v_norm, -1.0, 1.0))

    # Aspect Angle (AA) for enemy
    enm_v_norm = np.linalg.norm(enm_v)
    if enm_v_norm < 1e-6:
        aa = np.pi
    else:
        aa = np.arccos(np.clip(np.dot(enm_v, -los_vec) / enm_v_norm, -1.0, 1.0))
        
    # Advantage is high when our ATA is low (we are pointing at them)
    # and their AA is high (we are behind them)
    ata_adv = (np.pi - ata) / np.pi
    aa_adv = (np.pi - aa) / np.pi

    # Combine advantages (a simple way is to average them)
    # A more sophisticated model might use a non-linear combination
    angle_adv = (ata_adv + (1 - aa_adv)) / 2.0
    return angle_adv


def get_alt_adv(ego_state, enm_state, max_alt_diff=10000.0):
    """
    Calculate altitude advantage. Higher is better.
    """
    alt_diff = ego_state[Property.ALT.value] - enm_state[Property.ALT.value]
    # Normalize advantage to be between -1 and 1
    alt_adv = np.clip(alt_diff / max_alt_diff, -1.0, 1.0)
    # Rescale to [0, 1]
    return (alt_adv + 1) / 2.0


def get_vel_adv(ego_state, enm_state, max_vel_diff=500.0):
    """
    Calculate velocity advantage. Faster is better.
    """
    ego_v_mag = np.linalg.norm([ego_state[Property.VEL_U.value], ego_state[Property.VEL_V.value], ego_state[Property.VEL_W.value]])
    enm_v_mag = np.linalg.norm([enm_state[Property.VEL_U.value], enm_state[Property.VEL_V.value], enm_state[Property.VEL_W.value]])
    
    vel_diff = ego_v_mag - enm_v_mag
    # Normalize advantage to be between -1 and 1
    vel_adv = np.clip(vel_diff / max_vel_diff, -1.0, 1.0)
    # Rescale to [0, 1]
    return (vel_adv + 1) / 2.0 