import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import cv2
from baseballcv.functions import LoadTools
from ultralytics import YOLO
import pickle

load_tools = LoadTools()

r_pull_inside = pd.read_csv("right_inside_pull.csv")
r_pull_outside = pd.read_csv("right_outside_pull.csv")
l_pull_inside = pd.read_csv("left_inside_pull.csv")
l_pull_outside = pd.read_csv("left_outside_pull.csv")

#cropping video function

def download_crop_video(play_id, raw_dir, crop_dir, start, end, broadcast="home"):
    
    if broadcast == "home":
        url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}&videoType=HOME"
    
    elif broadcast == "away":
        url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}&videoType=AWAY"
        
    else:
        print("broadcast angle is not properly defined")
        output_path = None
        return output_path

    ################################
    #download the raw video
    ################################
    
    # get the HTML
    r = requests.get(url)
    r.raise_for_status()

    # parse HTML with BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    source_tag = soup.find("video").find("source")
    
    if not source_tag or not source_tag.get("src"):
#         print("No video for this angle")
#         flag = 1
        output_path = None
        return output_path
        
    video_url = source_tag["src"]

    # download video
    video_path = os.path.join(raw_dir, f"{play_id}.mp4")
    r_vid = requests.get(video_url, stream=True)
    r_vid.raise_for_status()

    #save to a raw video folder
    with open(video_path, "wb") as f:
        for chunk in r_vid.iter_content(chunk_size=1024):
            f.write(chunk)
                
    
    ################################
    #crop the video
    ################################
    start_ms = start
    end_ms = end
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert milliseconds → frame numbers
    start_frame = int(start_ms / 1000 * fps)
    end_frame =int(end_ms / 1000 * fps)

    # Output video writer
    output_path = os.path.join(crop_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_num <= end_frame:
            out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    
    print(f"Cropped video saved to {output_path}")
#     flag = 1
    output_path = output_path
    
    return output_path

#computer vision object detection function

def get_hitter_boxes(video_path, detector, tracker_type="CSRT"):
    """
    Detects hitter in the first frame and tracks them throughout the video using OpenCV tracker.

    Args:
        video_path (str): Path to video file.
        detector (YOLO): YOLO model for first-frame detection.
        tracker_type (str): OpenCV tracker type ("CSRT" by default).

    Returns:
        hitter_boxes_per_frame (list of lists): Each element is [x1, y1, x2, y2] for each frame.
        flag (int): 0 if successful, 1 if video should be skipped (no hitter first frame or can't read).
    """
    hitter_boxes_per_frame = []
    flag = 0

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read video {video_path}")
        flag = 1
        cap.release()
        return hitter_boxes_per_frame, flag

    # Detect hitter in first frame
    result = detector(frame)
    boxes = result[0].boxes
    names = result[0].names
    hitter_indices = [j for j, cls in enumerate(boxes.cls) if names[int(cls)] == "hitter"]

    if not hitter_indices:
        # No hitter in first frame → skip this video
        flag = 1
        cap.release()
        return hitter_boxes_per_frame, flag

    # Hitter detected, take first box
    hitter_box = boxes.xyxy[hitter_indices[0]].cpu().numpy()
    hitter_boxes_per_frame.append(hitter_box)

    # Initialize tracker
    x1, y1, x2, y2 = map(int, hitter_box)
    if tracker_type.upper() == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type.upper() == "KCF":
        tracker = cv2.TrackerKCF_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    # Track hitter for remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, tracked_box = tracker.update(frame)
        if not success:
            print(f"Tracker lost hitter in video {video_path}")
            break

        x, y, w, h = tracked_box
        hitter_boxes_per_frame.append([x, y, x + w, y + h])

    cap.release()
    return hitter_boxes_per_frame, flag

    
#right-handed batters pulling inside pitches
r_pull_inside = pd.read_csv("right_inside_pull.csv")

video_dir = 'right_inside_pull_raw'
output_dir = 'right_inside_pull_crop'
pose_dir = 'right_inside_pull_pose'

os.makedirs(video_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)

#get the right detector for pitcher hitter catcher detection
model_path = load_tools.load_model("phc_detector")
detector = YOLO(model_path)
pose_model = YOLO("yolo11s-pose.pt")

for idx, row in enumerate(r_pull_inside.itertuples(), 1):
    
    if row.play_guid == "d3003ff8-841f-46e4-b198-f25c4552d251":
        continue
    
    pose_path = os.path.join(pose_dir, f"{row.play_guid}.pkl")
    
    if os.path.exists(pose_path):
        print(f"Pose estimation already exists {row.play_guid}")
        print(f"{idx}/{len(r_pull_inside)} done")
    
    else:
    
        flag = 0

        #############################################
        #download and crop home broadcast angle
        #############################################
        print(f"home broadcast {row.play_guid}")
        output_path = download_crop_video(play_id=row.play_guid, raw_dir=video_dir, crop_dir=output_dir, 
                                          start=row.release_time, end=row.end_time, broadcast="home")

        #if there is a video for the home broadcast
        if output_path is not None:

            #do computer vision
            hitter_boxes_per_frame, flag = get_hitter_boxes(output_path, detector)

        #if there is no home broadcast feed or the previous for loop brok, get the away broadcast feed
        if output_path is None or flag == 1:

            #reset flag
            flag = 0
            
            print(f"away broadcast {row.play_guid}")
            output_path = download_crop_video(play_id=row.play_guid, raw_dir=video_dir, crop_dir=output_dir, 
                                          start=row.release_time, end=row.end_time, broadcast="away")

            #if there is no proper away feed break out of the iteration of the for loop
            if output_path is None:
                print(f"No Proper Video Available for {row.play_guid}")
                continue #shouldn't break the whole for loop, only the iteration

            #if path is good run computervision
            hitter_boxes_per_frame, flag = get_hitter_boxes(output_path, detector)

        if flag == 1:
            print(f"no good video for {row.play_guid}")
            continue

        #pose estimation
        # video_path = "video.mp4"
        cap = cv2.VideoCapture(output_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        pose_results = []  # store results per frame

        for frame_idx, box in enumerate(hitter_boxes_per_frame):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            x1, y1, x2, y2 = map(int, box)

            # Clamp coordinates inside frame
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid box for {row.play_guid} on frame {frame_idx}")
                continue

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                print(f"Empty ROI for {row.play_guid} on frame {frame_idx}")
                continue

            # Run pose estimation on ROI
            result = pose_model(roi, verbose=False)

            # Store results
            pose_results.append({
                "frame": frame_idx,
                "box": box.tolist() if not isinstance(box, list) else box,
                "keypoints": result[0].keypoints.xy.tolist()
            })

                

        #save off pose estimation
#         pose_path = os.path.join(pose_dir, f"{row.play_guid}.pkl")
        with open(pose_path, "wb") as f:
            pickle.dump(pose_results, f)
            
        print(f"pose estimation saved {row.play_guid}")
        print(f"{idx}/{len(r_pull_inside)} done")
