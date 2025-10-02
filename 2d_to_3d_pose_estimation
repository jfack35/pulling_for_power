import torch
import yaml
import pickle
import json
from collections import OrderedDict
import pandas as pd
import subprocess
from multiprocessing import Pool, cpu_count

#set your working directory to the motionbert folder if motionbert is cloaned
# cd "C:/Users/nymet/OneDrive/Syracuse/cv/motionbert"

### Right Inside Pull ###
# Load original checkpoint
ckpt_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_og.bin'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model_pos']

# Strip 'module.' prefix
clean_state_dict = OrderedDict()
for k, v in state_dict.items():
    clean_state_dict[k.replace('module.', '')] = v

# Replace the original state dict
checkpoint['model_pos'] = clean_state_dict

# Save cleaned checkpoint
torch.save(checkpoint, 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin')

r_pull_inside = pd.read_csv("../right_inside_pull.csv")
video_dir = 'right_inside_pull_crop'
pose_dir = 'right_inside_pull_pose'
output_path = 'right_inside_pull_motionbert'

# test = r_pull_inside.head(2)
for idx, row in enumerate(r_pull_inside.itertuples(), 1):
    
    motionbert_path = f"../{output_path}/{row.play_guid}"
    if os.path.exists(motionbert_path):
        print(f"Motionbert already exists {row.play_guid}")
        print(f"{idx}/{len(r_pull_inside)} done")
        continue        
    
    pose_path = f"../{pose_dir}/{row.play_guid}.pkl"
    
    if not os.path.exists(pose_path) or os.path.getsize(pose_path) == 0:
        print(f"2D Pose estimation missing or empty {row.play_guid}")
        print(f"{idx}/{len(r_pull_inside)} done")
        continue

    # Load 2D pose results
    with open(pose_path, "rb") as f:
        pose_results = pickle.load(f)
            
    # Prep JSON for MotionBERT
    num_joints = 17  # adjust if your model has a different number of joints
    processed = []
    for frame in pose_results:
        flat_kpts = []
        if frame['keypoints']:  # only if keypoints exist
            for x, y in frame['keypoints'][0]:  # [1, 17, 2]
                flat_kpts.extend([x, y, 1.0])
        else:
            # fill with zeros if no keypoints detected
            flat_kpts.extend([0.0, 0.0, 0.0] * num_joints)

        processed.append({
            'frame': frame['frame'],
            'idx': 0,  # single person
            'keypoints': flat_kpts
        })

    # Save JSON
    json_path = f"../{pose_dir}/{row.play_guid}.json"
    with open(json_path, "w") as f:
        json.dump(processed, f)

    # Run MotionBERT inference
    subprocess.run([
        "python", "infer_wild.py",
        "--vid_path", f"../{video_dir}/{row.play_guid}.mp4",
        "--json_path", json_path,
        "--out_path", motionbert_path
    ])
        
    print(f"motionbert saved {row.play_guid}")
    print(f"{idx}/{len(r_pull_inside)} done")


### Left Inside Pull ###
# Load original checkpoint
ckpt_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_og.bin'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model_pos']

# Strip 'module.' prefix
clean_state_dict = OrderedDict()
for k, v in state_dict.items():
    clean_state_dict[k.replace('module.', '')] = v

# Replace the original state dict
checkpoint['model_pos'] = clean_state_dict

# Save cleaned checkpoint
torch.save(checkpoint, 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin')

l_pull_inside = pd.read_csv("../left_inside_pull.csv")
video_dir = 'left_inside_pull_crop'
pose_dir = 'left_inside_pull_pose'
output_path = 'left_inside_pull_motionbert'

# test = r_pull_inside.head(2)
for idx, row in enumerate(l_pull_inside.itertuples(), 1):
    
    motionbert_path = f"../{output_path}/{row.play_guid}"
    if os.path.exists(motionbert_path):
        print(f"Motionbert already exists {row.play_guid}")
        print(f"{idx}/{len(l_pull_inside)} done")
        continue        
    
    pose_path = f"../{pose_dir}/{row.play_guid}.pkl"
    
    if not os.path.exists(pose_path) or os.path.getsize(pose_path) == 0:
        print(f"2D Pose estimation missing or empty {row.play_guid}")
        print(f"{idx}/{len(l_pull_inside)} done")
        continue

    # Load 2D pose results
    with open(pose_path, "rb") as f:
        pose_results = pickle.load(f)
            
    # Prep JSON for MotionBERT
    num_joints = 17  # adjust if your model has a different number of joints
    processed = []
    for frame in pose_results:
        flat_kpts = []
        if frame['keypoints']:  # only if keypoints exist
            for x, y in frame['keypoints'][0]:  # [1, 17, 2]
                flat_kpts.extend([x, y, 1.0])
        else:
            # fill with zeros if no keypoints detected
            flat_kpts.extend([0.0, 0.0, 0.0] * num_joints)

        processed.append({
            'frame': frame['frame'],
            'idx': 0,  # single person
            'keypoints': flat_kpts
        })

    # Save JSON
    json_path = f"../{pose_dir}/{row.play_guid}.json"
    with open(json_path, "w") as f:
        json.dump(processed, f)

    # Run MotionBERT inference
    subprocess.run([
        "python", "infer_wild.py",
        "--vid_path", f"../{video_dir}/{row.play_guid}.mp4",
        "--json_path", json_path,
        "--out_path", motionbert_path
    ])
        
    print(f"motionbert saved {row.play_guid}")
    print(f"{idx}/{len(l_pull_inside)} done")


### Right Outside Pull ###
# Load original checkpoint
ckpt_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_og.bin'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model_pos']

# Strip 'module.' prefix
clean_state_dict = OrderedDict()
for k, v in state_dict.items():
    clean_state_dict[k.replace('module.', '')] = v

# Replace the original state dict
checkpoint['model_pos'] = clean_state_dict

# Save cleaned checkpoint
torch.save(checkpoint, 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin')

r_pull_outside = pd.read_csv("../right_outside_pull.csv")
video_dir = 'right_outside_pull_crop'
pose_dir = 'right_outside_pull_pose'
output_path = 'right_outside_pull_motionbert'

# test = r_pull_inside.head(2)
for idx, row in enumerate(r_pull_outside.itertuples(), 1):
    
    motionbert_path = f"../{output_path}/{row.play_guid}"
    if os.path.exists(motionbert_path):
        print(f"Motionbert already exists {row.play_guid}")
        print(f"{idx}/{len(r_pull_outside)} done")
        continue        
    
    pose_path = f"../{pose_dir}/{row.play_guid}.pkl"
    
    if not os.path.exists(pose_path) or os.path.getsize(pose_path) == 0:
        print(f"2D Pose estimation missing or empty {row.play_guid}")
        print(f"{idx}/{len(r_pull_outside)} done")
        continue

    # Load 2D pose results
    with open(pose_path, "rb") as f:
        pose_results = pickle.load(f)
            
    # Prep JSON for MotionBERT
    num_joints = 17  # adjust if your model has a different number of joints
    processed = []
    for frame in pose_results:
        flat_kpts = []
        if frame['keypoints']:  # only if keypoints exist
            for x, y in frame['keypoints'][0]:  # [1, 17, 2]
                flat_kpts.extend([x, y, 1.0])
        else:
            # fill with zeros if no keypoints detected
            flat_kpts.extend([0.0, 0.0, 0.0] * num_joints)

        processed.append({
            'frame': frame['frame'],
            'idx': 0,  # single person
            'keypoints': flat_kpts
        })

    # Save JSON
    json_path = f"../{pose_dir}/{row.play_guid}.json"
    with open(json_path, "w") as f:
        json.dump(processed, f)

    # Run MotionBERT inference
    subprocess.run([
        "python", "infer_wild.py",
        "--vid_path", f"../{video_dir}/{row.play_guid}.mp4",
        "--json_path", json_path,
        "--out_path", motionbert_path
    ])
        
    print(f"motionbert saved {row.play_guid}")
    print(f"{idx}/{len(r_pull_outside)} done")

    ### Left Outside Pull ###
    # Load original checkpoint
ckpt_path = 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_og.bin'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model_pos']

# Strip 'module.' prefix
clean_state_dict = OrderedDict()
for k, v in state_dict.items():
    clean_state_dict[k.replace('module.', '')] = v

# Replace the original state dict
checkpoint['model_pos'] = clean_state_dict

# Save cleaned checkpoint
torch.save(checkpoint, 'checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin')

l_pull_outside = pd.read_csv("../left_outside_pull.csv")
video_dir = 'left_outside_pull_crop'
pose_dir = 'left_outside_pull_pose'
output_path = 'left_outside_pull_motionbert'

# test = r_pull_inside.head(2)
for idx, row in enumerate(l_pull_outside.itertuples(), 1):
    
    motionbert_path = f"../{output_path}/{row.play_guid}"
    if os.path.exists(motionbert_path):
        print(f"Motionbert already exists {row.play_guid}")
        print(f"{idx}/{len(l_pull_outside)} done")
        continue        
    
    pose_path = f"../{pose_dir}/{row.play_guid}.pkl"
    
    if not os.path.exists(pose_path) or os.path.getsize(pose_path) == 0:
        print(f"2D Pose estimation missing or empty {row.play_guid}")
        print(f"{idx}/{len(l_pull_outside)} done")
        continue

    # Load 2D pose results
    with open(pose_path, "rb") as f:
        pose_results = pickle.load(f)
            
    # Prep JSON for MotionBERT
    num_joints = 17  # adjust if your model has a different number of joints
    processed = []
    for frame in pose_results:
        flat_kpts = []
        if frame['keypoints']:  # only if keypoints exist
            for x, y in frame['keypoints'][0]:  # [1, 17, 2]
                flat_kpts.extend([x, y, 1.0])
        else:
            # fill with zeros if no keypoints detected
            flat_kpts.extend([0.0, 0.0, 0.0] * num_joints)

        processed.append({
            'frame': frame['frame'],
            'idx': 0,  # single person
            'keypoints': flat_kpts
        })

    # Save JSON
    json_path = f"../{pose_dir}/{row.play_guid}.json"
    with open(json_path, "w") as f:
        json.dump(processed, f)

    # Run MotionBERT inference
    subprocess.run([
        "python", "infer_wild.py",
        "--vid_path", f"../{video_dir}/{row.play_guid}.mp4",
        "--json_path", json_path,
        "--out_path", motionbert_path
    ])
        
    print(f"motionbert saved {row.play_guid}")
    print(f"{idx}/{len(l_pull_outside)} done")

    
