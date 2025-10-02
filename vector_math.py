!pip install pyarrow

import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

H36M_IDX = {
    "Pelvis": 0,       # root
    "R_Hip": 4,
    "R_Knee": 5,
    "R_Ankle": 6,
    "L_Hip": 1,
    "L_Knee": 2,
    "L_Ankle": 3,
    "Spine": 7,
    "Thorax": 8,
    "Neck": 9,
    "Head": 10,
    "L_Shoulder": 14,
    "L_Elbow": 15,
    "L_Wrist": 16,
    "R_Shoulder": 11,
    "R_Elbow": 12,
    "R_Wrist": 13
}
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter with zero-phase filtering.

    Parameters
    ----------
    data : np.ndarray
        Input signal (can be 1D, 2D, or 3D — filtering happens along axis=0).
    cutoff : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int, optional
        Filter order (default=4).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.
    """
    nyq = 0.5 * fs                      # Nyquist frequency
    normal_cutoff = cutoff / nyq        # Normalized cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter along axis=0 (time dimension), preserve other dims
    return filtfilt(b, a, data, axis=0)

def angle_between(v1, v2):
  v1_norm = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
  v2_norm = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
  dot = np.sum(v1_norm * v2_norm, axis=-1)
  return np.arccos(np.clip(dot, -1.0, 1.0))  # radians

### Right Pull Inside ###
r_pull_inside = pd.read_csv("right_inside_pull.csv")
pose_dir = 'right_inside_pull_motionbert'

output_dir = 'right_inside_pull_ik'
os.makedirs(output_dir, exist_ok=True)

fps = 60
dt = 1/fps
cutoff=10

all_dfs = []

# test = r_pull_inside.head(5)
for idx, row in enumerate(r_pull_inside.itertuples(), 1):

        pose_path = f"{pose_dir}/{row.play_guid}/X3D.npy"

        if not os.path.exists(pose_path):
            print(f"no 3d pose for this play {row.play_guid}")
            continue
        
        else: 
            sample_3d = np.load(f"{pose_dir}/{row.play_guid}/X3D.npy")
            
            if sample_3d.shape[0] <= 15:
                continue
            
            vel = butter_lowpass_filter((np.gradient(sample_3d, dt, axis=0)), cutoff=cutoff, fs=fps)    # shape (T, J, 3)
            acc = butter_lowpass_filter((np.gradient(vel, dt, axis=0)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Wrist
            ###########################
            
            #left wrist
            left_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            #right wrist
            right_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            ###########################
            # Elbow
            ###########################
            
            #left elbow angle
            shoulder = sample_3d[:, 14]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 15]
            wrist    = sample_3d[:, 16]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            left_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            left_elbow_ang_velo = butter_lowpass_filter((np.gradient(left_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            left_elbow_ang_acc = butter_lowpass_filter((np.gradient(left_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right elbow angle
            shoulder = sample_3d[:, 11]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 12]
            wrist    = sample_3d[:, 13]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            right_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)  # shape (T, 3)
            right_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_elbow_ang_velo = butter_lowpass_filter((np.gradient(right_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            right_elbow_ang_acc = butter_lowpass_filter((np.gradient(right_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Knee
            ###########################
            
            #left knee angle
            hip = sample_3d[:, 1]  # index depends on H36M joint mapping
            knee = sample_3d[:, 2]
            ankle = sample_3d[:, 3]

            vec1 = hip - knee
            vec2 = ankle - knee
            left_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)

            left_knee_ang_velo = butter_lowpass_filter((np.gradient(left_knee_angle, dt)), cutoff=cutoff, fs=fps)
            left_knee_ang_acc = butter_lowpass_filter((np.gradient(left_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right knee angle
            hip = sample_3d[:, 4]  # index depends on H36M joint mapping
            knee = sample_3d[:, 5]
            ankle = sample_3d[:, 6]

            vec1 = hip - knee
            vec2 = ankle - knee
            right_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_knee_ang_velo = butter_lowpass_filter((np.gradient(right_knee_angle, dt)), cutoff=cutoff, fs=fps)
            right_knee_ang_acc = butter_lowpass_filter((np.gradient(right_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Pelvis
            ###########################
            
            #pelvic stuff
            left_hip = sample_3d[:, 1]
            right_hip = sample_3d[:, 4]
            pelvis_vec = right_hip - left_hip

            # Project to horizontal plane
            pelvis_vec[:, 2] = 0
            pelvis_angle = butter_lowpass_filter((np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))), cutoff=cutoff, fs=fps)  # shape (T,)
            pelvis_ang_velo = butter_lowpass_filter((np.gradient(pelvis_angle, dt)), cutoff=cutoff, fs=fps)
            pelvis_ang_acc = butter_lowpass_filter((np.gradient(pelvis_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #saving it off
            T = sample_3d.shape[0]
            pguid = row.play_guid

            def make_scalar_df(data, joint, feature, segment):
                return pd.DataFrame({
                    "frame": np.arange(T),
                    "bat_side": "R",
                    "player": row.batter_id,
                    "horizontal": "inside",
                    "spray": "pull",
                    "joint": joint,
                    "feature_type": feature,
                    "axis": None,
                    "value": data,
                    "play_guid": pguid,
                    "segment": segment
                })

            def make_vector_df(data, joint, feature, segment):
                dfs = []
                if data.ndim != 2 or data.shape[1] != 3:
                    raise ValueError(f"Expected shape (T, 3) for {joint} {feature}, got {data.shape}")

                T = data.shape[0]
                for i, axis in enumerate(["x", "y", "z"]):
                    df = pd.DataFrame({
                        "frame": np.arange(T),
                        "bat_side": "R",
                        "player": row.batter_id,
                        "horizontal": "inside",
                        "spray": "pull",
                        "joint": joint,
                        "feature_type": feature,
                        "axis": axis,
                        "value": data[:, i],
                        "play_guid": pguid,
                        "segment": segment
                    })
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)

            # Wrist
            all_dfs.append(make_vector_df(left_wrist_velo, "L_Wrist", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_wrist_acc, "L_Wrist", "acceleration", "left_arm"))
            all_dfs.append(make_vector_df(right_wrist_velo, "R_Wrist", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_wrist_acc, "R_Wrist", "acceleration", "right_arm"))

            # Elbow
            all_dfs.append(make_scalar_df(left_elbow_angle, "L_Elbow", "angle", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_velo, "L_Elbow", "angular_velocity", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_acc, "L_Elbow", "angular_acceleration", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_velo, "L_Elbow", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_acc, "L_Elbow", "acceleration", "left_arm"))

            all_dfs.append(make_scalar_df(right_elbow_angle, "R_Elbow", "angle", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_velo, "R_Elbow", "angular_velocity", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_acc, "R_Elbow", "angular_acceleration", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_velo, "R_Elbow", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_acc, "R_Elbow", "acceleration", "right_arm"))

            # Knee
            all_dfs.append(make_scalar_df(left_knee_angle, "L_Knee", "angle", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_velo, "L_Knee", "angular_velocity", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_acc, "L_Knee", "angular_acceleration", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_velo, "L_Knee", "velocity", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_acc, "L_Knee", "acceleration", "left_leg"))

            all_dfs.append(make_scalar_df(right_knee_angle, "R_Knee", "angle", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_velo, "R_Knee", "angular_velocity", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_acc, "R_Knee", "angular_acceleration", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_velo, "R_Knee", "velocity", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_acc, "R_Knee", "acceleration", "right_leg"))

            # Pelvis
            all_dfs.append(make_scalar_df(pelvis_angle, "Pelvis", "angle", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_velo, "Pelvis", "angular_velocity", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_acc, "Pelvis", "angular_acceleration", "pelvis_rotation"))
            
            print(f"{idx}/{len(r_pull_inside)} done {row.play_guid}")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_parquet(os.path.join(output_dir, "right_inside_pull_ik_math.parquet"), index=False)

### Right Pull Outside ###
r_pull_outside = pd.read_csv("right_outside_pull.csv")
pose_dir = 'right_outside_pull_motionbert'

output_dir = 'right_outside_pull_ik'
os.makedirs(output_dir, exist_ok=True)

fps = 60
dt = 1/fps
cutoff=10

all_dfs = []

# test = r_pull_inside.head(5)
for idx, row in enumerate(r_pull_outside.itertuples(), 1):

        pose_path = f"{pose_dir}/{row.play_guid}/X3D.npy"

        if not os.path.exists(pose_path):
            print(f"no 3d pose for this play {row.play_guid}")
            continue
        
        else: 
            sample_3d = np.load(f"{pose_dir}/{row.play_guid}/X3D.npy")
            
            if sample_3d.shape[0] <= 15:
                continue
            
            vel = butter_lowpass_filter((np.gradient(sample_3d, dt, axis=0)), cutoff=cutoff, fs=fps)    # shape (T, J, 3)
            acc = butter_lowpass_filter((np.gradient(vel, dt, axis=0)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Wrist
            ###########################
            
            #left wrist
            left_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            #right wrist
            right_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            ###########################
            # Elbow
            ###########################
            
            #left elbow angle
            shoulder = sample_3d[:, 14]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 15]
            wrist    = sample_3d[:, 16]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            left_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            left_elbow_ang_velo = butter_lowpass_filter((np.gradient(left_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            left_elbow_ang_acc = butter_lowpass_filter((np.gradient(left_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right elbow angle
            shoulder = sample_3d[:, 11]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 12]
            wrist    = sample_3d[:, 13]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            right_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)  # shape (T, 3)
            right_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_elbow_ang_velo = butter_lowpass_filter((np.gradient(right_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            right_elbow_ang_acc = butter_lowpass_filter((np.gradient(right_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Knee
            ###########################
            
            #left knee angle
            hip = sample_3d[:, 1]  # index depends on H36M joint mapping
            knee = sample_3d[:, 2]
            ankle = sample_3d[:, 3]

            vec1 = hip - knee
            vec2 = ankle - knee
            left_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)

            left_knee_ang_velo = butter_lowpass_filter((np.gradient(left_knee_angle, dt)), cutoff=cutoff, fs=fps)
            left_knee_ang_acc = butter_lowpass_filter((np.gradient(left_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right knee angle
            hip = sample_3d[:, 4]  # index depends on H36M joint mapping
            knee = sample_3d[:, 5]
            ankle = sample_3d[:, 6]

            vec1 = hip - knee
            vec2 = ankle - knee
            right_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_knee_ang_velo = butter_lowpass_filter((np.gradient(right_knee_angle, dt)), cutoff=cutoff, fs=fps)
            right_knee_ang_acc = butter_lowpass_filter((np.gradient(right_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Pelvis
            ###########################
            
            #pelvic stuff
            left_hip = sample_3d[:, 1]
            right_hip = sample_3d[:, 4]
            pelvis_vec = right_hip - left_hip

            # Project to horizontal plane
            pelvis_vec[:, 2] = 0
            pelvis_angle = butter_lowpass_filter((np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))), cutoff=cutoff, fs=fps)  # shape (T,)
            pelvis_ang_velo = butter_lowpass_filter((np.gradient(pelvis_angle, dt)), cutoff=cutoff, fs=fps)
            pelvis_ang_acc = butter_lowpass_filter((np.gradient(pelvis_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #saving it off
            T = sample_3d.shape[0]
            pguid = row.play_guid

            def make_scalar_df(data, joint, feature, segment):
                return pd.DataFrame({
                    "frame": np.arange(T),
                    "bat_side": "R",
                    "player": row.batter_id,
                    "horizontal": "outside",
                    "spray": "pull",
                    "joint": joint,
                    "feature_type": feature,
                    "axis": None,
                    "value": data,
                    "play_guid": pguid,
                    "segment": segment
                })

            def make_vector_df(data, joint, feature, segment):
                dfs = []
                if data.ndim != 2 or data.shape[1] != 3:
                    raise ValueError(f"Expected shape (T, 3) for {joint} {feature}, got {data.shape}")

                T = data.shape[0]
                for i, axis in enumerate(["x", "y", "z"]):
                    df = pd.DataFrame({
                        "frame": np.arange(T),
                        "bat_side": "R",
                        "player": row.batter_id,
                        "horizontal": "outside",
                        "spray": "pull",
                        "joint": joint,
                        "feature_type": feature,
                        "axis": axis,
                        "value": data[:, i],
                        "play_guid": pguid,
                        "segment": segment
                    })
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)

            # Wrist
            all_dfs.append(make_vector_df(left_wrist_velo, "L_Wrist", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_wrist_acc, "L_Wrist", "acceleration", "left_arm"))
            all_dfs.append(make_vector_df(right_wrist_velo, "R_Wrist", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_wrist_acc, "R_Wrist", "acceleration", "right_arm"))

            # Elbow
            all_dfs.append(make_scalar_df(left_elbow_angle, "L_Elbow", "angle", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_velo, "L_Elbow", "angular_velocity", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_acc, "L_Elbow", "angular_acceleration", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_velo, "L_Elbow", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_acc, "L_Elbow", "acceleration", "left_arm"))

            all_dfs.append(make_scalar_df(right_elbow_angle, "R_Elbow", "angle", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_velo, "R_Elbow", "angular_velocity", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_acc, "R_Elbow", "angular_acceleration", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_velo, "R_Elbow", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_acc, "R_Elbow", "acceleration", "right_arm"))

            # Knee
            all_dfs.append(make_scalar_df(left_knee_angle, "L_Knee", "angle", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_velo, "L_Knee", "angular_velocity", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_acc, "L_Knee", "angular_acceleration", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_velo, "L_Knee", "velocity", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_acc, "L_Knee", "acceleration", "left_leg"))

            all_dfs.append(make_scalar_df(right_knee_angle, "R_Knee", "angle", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_velo, "R_Knee", "angular_velocity", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_acc, "R_Knee", "angular_acceleration", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_velo, "R_Knee", "velocity", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_acc, "R_Knee", "acceleration", "right_leg"))

            # Pelvis
            all_dfs.append(make_scalar_df(pelvis_angle, "Pelvis", "angle", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_velo, "Pelvis", "angular_velocity", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_acc, "Pelvis", "angular_acceleration", "pelvis_rotation"))
            
            print(f"{idx}/{len(r_pull_outside)} done {row.play_guid}")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_parquet(os.path.join(output_dir, "right_outside_pull_ik_math.parquet"), index=False)

### Left Pull Inside ###
l_pull_inside = pd.read_csv("left_inside_pull.csv")
pose_dir = 'left_inside_pull_motionbert'

output_dir = 'left_inside_pull_ik'
os.makedirs(output_dir, exist_ok=True)

fps = 60
dt = 1/fps
cutoff=10

all_dfs = []

# test = r_pull_inside.head(5)
for idx, row in enumerate(l_pull_inside.itertuples(), 1):

        pose_path = f"{pose_dir}/{row.play_guid}/X3D.npy"

        if not os.path.exists(pose_path):
            print(f"no 3d pose for this play {row.play_guid}")
            continue
        
        else: 
            sample_3d = np.load(f"{pose_dir}/{row.play_guid}/X3D.npy")
            
            if sample_3d.shape[0] <= 15:
                continue
            
            vel = butter_lowpass_filter((np.gradient(sample_3d, dt, axis=0)), cutoff=cutoff, fs=fps)    # shape (T, J, 3)
            acc = butter_lowpass_filter((np.gradient(vel, dt, axis=0)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Wrist
            ###########################
            
            #left wrist
            left_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            #right wrist
            right_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            ###########################
            # Elbow
            ###########################
            
            #left elbow angle
            shoulder = sample_3d[:, 14]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 15]
            wrist    = sample_3d[:, 16]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            left_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            left_elbow_ang_velo = butter_lowpass_filter((np.gradient(left_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            left_elbow_ang_acc = butter_lowpass_filter((np.gradient(left_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right elbow angle
            shoulder = sample_3d[:, 11]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 12]
            wrist    = sample_3d[:, 13]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            right_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)  # shape (T, 3)
            right_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_elbow_ang_velo = butter_lowpass_filter((np.gradient(right_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            right_elbow_ang_acc = butter_lowpass_filter((np.gradient(right_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Knee
            ###########################
            
            #left knee angle
            hip = sample_3d[:, 1]  # index depends on H36M joint mapping
            knee = sample_3d[:, 2]
            ankle = sample_3d[:, 3]

            vec1 = hip - knee
            vec2 = ankle - knee
            left_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)

            left_knee_ang_velo = butter_lowpass_filter((np.gradient(left_knee_angle, dt)), cutoff=cutoff, fs=fps)
            left_knee_ang_acc = butter_lowpass_filter((np.gradient(left_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right knee angle
            hip = sample_3d[:, 4]  # index depends on H36M joint mapping
            knee = sample_3d[:, 5]
            ankle = sample_3d[:, 6]

            vec1 = hip - knee
            vec2 = ankle - knee
            right_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_knee_ang_velo = butter_lowpass_filter((np.gradient(right_knee_angle, dt)), cutoff=cutoff, fs=fps)
            right_knee_ang_acc = butter_lowpass_filter((np.gradient(right_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Pelvis
            ###########################
            
            #pelvic stuff
            left_hip = sample_3d[:, 1]
            right_hip = sample_3d[:, 4]
            pelvis_vec = right_hip - left_hip

            # Project to horizontal plane
            pelvis_vec[:, 2] = 0
            pelvis_angle = butter_lowpass_filter((np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))), cutoff=cutoff, fs=fps)  # shape (T,)
            pelvis_ang_velo = butter_lowpass_filter((np.gradient(pelvis_angle, dt)), cutoff=cutoff, fs=fps)
            pelvis_ang_acc = butter_lowpass_filter((np.gradient(pelvis_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #saving it off
            T = sample_3d.shape[0]
            pguid = row.play_guid

            def make_scalar_df(data, joint, feature, segment):
                return pd.DataFrame({
                    "frame": np.arange(T),
                    "bat_side": "L",
                    "player": row.batter_id,
                    "horizontal": "inside",
                    "spray": "pull",
                    "joint": joint,
                    "feature_type": feature,
                    "axis": None,
                    "value": data,
                    "play_guid": pguid,
                    "segment": segment
                })

            def make_vector_df(data, joint, feature, segment):
                dfs = []
                if data.ndim != 2 or data.shape[1] != 3:
                    raise ValueError(f"Expected shape (T, 3) for {joint} {feature}, got {data.shape}")

                T = data.shape[0]
                for i, axis in enumerate(["x", "y", "z"]):
                    df = pd.DataFrame({
                        "frame": np.arange(T),
                        "bat_side": "L",
                        "player": row.batter_id,
                        "horizontal": "inside",
                        "spray": "pull",
                        "joint": joint,
                        "feature_type": feature,
                        "axis": axis,
                        "value": data[:, i],
                        "play_guid": pguid,
                        "segment": segment
                    })
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)

            # Wrist
            all_dfs.append(make_vector_df(left_wrist_velo, "L_Wrist", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_wrist_acc, "L_Wrist", "acceleration", "left_arm"))
            all_dfs.append(make_vector_df(right_wrist_velo, "R_Wrist", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_wrist_acc, "R_Wrist", "acceleration", "right_arm"))

            # Elbow
            all_dfs.append(make_scalar_df(left_elbow_angle, "L_Elbow", "angle", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_velo, "L_Elbow", "angular_velocity", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_acc, "L_Elbow", "angular_acceleration", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_velo, "L_Elbow", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_acc, "L_Elbow", "acceleration", "left_arm"))

            all_dfs.append(make_scalar_df(right_elbow_angle, "R_Elbow", "angle", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_velo, "R_Elbow", "angular_velocity", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_acc, "R_Elbow", "angular_acceleration", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_velo, "R_Elbow", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_acc, "R_Elbow", "acceleration", "right_arm"))

            # Knee
            all_dfs.append(make_scalar_df(left_knee_angle, "L_Knee", "angle", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_velo, "L_Knee", "angular_velocity", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_acc, "L_Knee", "angular_acceleration", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_velo, "L_Knee", "velocity", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_acc, "L_Knee", "acceleration", "left_leg"))

            all_dfs.append(make_scalar_df(right_knee_angle, "R_Knee", "angle", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_velo, "R_Knee", "angular_velocity", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_acc, "R_Knee", "angular_acceleration", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_velo, "R_Knee", "velocity", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_acc, "R_Knee", "acceleration", "right_leg"))

            # Pelvis
            all_dfs.append(make_scalar_df(pelvis_angle, "Pelvis", "angle", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_velo, "Pelvis", "angular_velocity", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_acc, "Pelvis", "angular_acceleration", "pelvis_rotation"))
            
            print(f"{idx}/{len(l_pull_inside)} done {row.play_guid}")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_parquet(os.path.join(output_dir, "left_inside_pull_ik_math.parquet"), index=False)

### Left Pull Outside ###
l_pull_outside = pd.read_csv("left_outside_pull.csv")
pose_dir = 'left_outside_pull_motionbert'

output_dir = 'left_outside_pull_ik'
os.makedirs(output_dir, exist_ok=True)

fps = 60
dt = 1/fps
cutoff=10

all_dfs = []

# test = r_pull_inside.head(5)
for idx, row in enumerate(l_pull_outside.itertuples(), 1):

        pose_path = f"{pose_dir}/{row.play_guid}/X3D.npy"

        if not os.path.exists(pose_path):
            print(f"no 3d pose for this play {row.play_guid}")
            continue
        
        else: 
            sample_3d = np.load(f"{pose_dir}/{row.play_guid}/X3D.npy")
            
            if sample_3d.shape[0] <= 15:
                continue
            
            vel = butter_lowpass_filter((np.gradient(sample_3d, dt, axis=0)), cutoff=cutoff, fs=fps)    # shape (T, J, 3)
            acc = butter_lowpass_filter((np.gradient(vel, dt, axis=0)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Wrist
            ###########################
            
            #left wrist
            left_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            #right wrist
            right_wrist_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_wrist_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Wrist'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            ###########################
            # Elbow
            ###########################
            
            #left elbow angle
            shoulder = sample_3d[:, 14]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 15]
            wrist    = sample_3d[:, 16]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            left_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            left_elbow_ang_velo = butter_lowpass_filter((np.gradient(left_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            left_elbow_ang_acc = butter_lowpass_filter((np.gradient(left_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right elbow angle
            shoulder = sample_3d[:, 11]  # index depends on H36M joint mapping
            elbow    = sample_3d[:, 12]
            wrist    = sample_3d[:, 13]

            vec1 = shoulder - elbow
            vec2 = wrist - elbow
            right_elbow_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_elbow_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)  # shape (T, 3)
            right_elbow_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Elbow'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_elbow_ang_velo = butter_lowpass_filter((np.gradient(right_elbow_angle, dt)), cutoff=cutoff, fs=fps)
            right_elbow_ang_acc = butter_lowpass_filter((np.gradient(right_elbow_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Knee
            ###########################
            
            #left knee angle
            hip = sample_3d[:, 1]  # index depends on H36M joint mapping
            knee = sample_3d[:, 2]
            ankle = sample_3d[:, 3]

            vec1 = hip - knee
            vec2 = ankle - knee
            left_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            left_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            left_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['L_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)

            left_knee_ang_velo = butter_lowpass_filter((np.gradient(left_knee_angle, dt)), cutoff=cutoff, fs=fps)
            left_knee_ang_acc = butter_lowpass_filter((np.gradient(left_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #right knee angle
            hip = sample_3d[:, 4]  # index depends on H36M joint mapping
            knee = sample_3d[:, 5]
            ankle = sample_3d[:, 6]

            vec1 = hip - knee
            vec2 = ankle - knee
            right_knee_angle = butter_lowpass_filter((np.degrees(angle_between(vec1, vec2))), cutoff=cutoff, fs=fps)  # shape (T,)
            right_knee_velo = butter_lowpass_filter((vel[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            right_knee_acc = butter_lowpass_filter((acc[:, H36M_IDX['R_Knee'], :]), cutoff=cutoff, fs=fps)   # shape (T, 3)
            
            right_knee_ang_velo = butter_lowpass_filter((np.gradient(right_knee_angle, dt)), cutoff=cutoff, fs=fps)
            right_knee_ang_acc = butter_lowpass_filter((np.gradient(right_knee_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            ###########################
            # Pelvis
            ###########################
            
            #pelvic stuff
            left_hip = sample_3d[:, 1]
            right_hip = sample_3d[:, 4]
            pelvis_vec = right_hip - left_hip

            # Project to horizontal plane
            pelvis_vec[:, 2] = 0
            pelvis_angle = butter_lowpass_filter((np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))), cutoff=cutoff, fs=fps)  # shape (T,)
            pelvis_ang_velo = butter_lowpass_filter((np.gradient(pelvis_angle, dt)), cutoff=cutoff, fs=fps)
            pelvis_ang_acc = butter_lowpass_filter((np.gradient(pelvis_ang_velo, dt)), cutoff=cutoff, fs=fps)
            
            #saving it off
            T = sample_3d.shape[0]
            pguid = row.play_guid

            def make_scalar_df(data, joint, feature, segment):
                return pd.DataFrame({
                    "frame": np.arange(T),
                    "bat_side": "L",
                    "player": row.batter_id,
                    "horizontal": "outside",
                    "spray": "pull",
                    "joint": joint,
                    "feature_type": feature,
                    "axis": None,
                    "value": data,
                    "play_guid": pguid,
                    "segment": segment
                })

            def make_vector_df(data, joint, feature, segment):
                dfs = []
                if data.ndim != 2 or data.shape[1] != 3:
                    raise ValueError(f"Expected shape (T, 3) for {joint} {feature}, got {data.shape}")

                T = data.shape[0]
                for i, axis in enumerate(["x", "y", "z"]):
                    df = pd.DataFrame({
                        "frame": np.arange(T),
                        "bat_side": "L",
                        "player": row.batter_id,
                        "horizontal": "outside",
                        "spray": "pull",
                        "joint": joint,
                        "feature_type": feature,
                        "axis": axis,
                        "value": data[:, i],
                        "play_guid": pguid,
                        "segment": segment
                    })
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)

            # Wrist
            all_dfs.append(make_vector_df(left_wrist_velo, "L_Wrist", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_wrist_acc, "L_Wrist", "acceleration", "left_arm"))
            all_dfs.append(make_vector_df(right_wrist_velo, "R_Wrist", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_wrist_acc, "R_Wrist", "acceleration", "right_arm"))

            # Elbow
            all_dfs.append(make_scalar_df(left_elbow_angle, "L_Elbow", "angle", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_velo, "L_Elbow", "angular_velocity", "left_arm"))
            all_dfs.append(make_scalar_df(left_elbow_ang_acc, "L_Elbow", "angular_acceleration", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_velo, "L_Elbow", "velocity", "left_arm"))
            all_dfs.append(make_vector_df(left_elbow_acc, "L_Elbow", "acceleration", "left_arm"))

            all_dfs.append(make_scalar_df(right_elbow_angle, "R_Elbow", "angle", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_velo, "R_Elbow", "angular_velocity", "right_arm"))
            all_dfs.append(make_scalar_df(right_elbow_ang_acc, "R_Elbow", "angular_acceleration", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_velo, "R_Elbow", "velocity", "right_arm"))
            all_dfs.append(make_vector_df(right_elbow_acc, "R_Elbow", "acceleration", "right_arm"))

            # Knee
            all_dfs.append(make_scalar_df(left_knee_angle, "L_Knee", "angle", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_velo, "L_Knee", "angular_velocity", "left_leg"))
            all_dfs.append(make_scalar_df(left_knee_ang_acc, "L_Knee", "angular_acceleration", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_velo, "L_Knee", "velocity", "left_leg"))
            all_dfs.append(make_vector_df(left_knee_acc, "L_Knee", "acceleration", "left_leg"))

            all_dfs.append(make_scalar_df(right_knee_angle, "R_Knee", "angle", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_velo, "R_Knee", "angular_velocity", "right_leg"))
            all_dfs.append(make_scalar_df(right_knee_ang_acc, "R_Knee", "angular_acceleration", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_velo, "R_Knee", "velocity", "right_leg"))
            all_dfs.append(make_vector_df(right_knee_acc, "R_Knee", "acceleration", "right_leg"))

            # Pelvis
            all_dfs.append(make_scalar_df(pelvis_angle, "Pelvis", "angle", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_velo, "Pelvis", "angular_velocity", "pelvis_rotation"))
            all_dfs.append(make_scalar_df(pelvis_ang_acc, "Pelvis", "angular_acceleration", "pelvis_rotation"))
            
            print(f"{idx}/{len(l_pull_outside)} done {row.play_guid}")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_parquet(os.path.join(output_dir, "left_outside_pull_ik_math.parquet"), index=False)



  
