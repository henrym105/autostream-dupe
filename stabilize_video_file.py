import cv2
import numpy as np
import os
from src.constants import (
    CUR_DIR
)

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print(f"Number of Frames: {n_frames}")
    print(f"Frame Width: {w}, Frame Height: {h}")
    print(f"FPS: {fps}")

    # Read the first frame
    _, prev = cap.read()
    if prev is None:
        print("Error: Unable to read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames-1):
        success, curr = cap.read()
        if not success:
            print(f"Failed to read frame {i}")
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    print("Transforms calculated")

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    print("Trajectory smoothed")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if original_fourcc == 0 else original_fourcc
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in range(n_frames-1):
        success, frame = cap.read()
        if not success:
            print(f"Failed to read frame {i} during stabilization")
            break
        dx, dy, da = transforms_smooth[i]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        out.write(frame_stabilized)

        # Debugging: Save intermediate frames
        if i % 100 == 0:
            debug_frame_path = os.path.join(CUR_DIR, "data", "debug", f"frame_{i}.jpg")
            cv2.imwrite(debug_frame_path, frame_stabilized)
            print(f"Saved debug frame {i} to {debug_frame_path}")

    print("Video stabilization complete")

    cap.release()
    out.release()

def smooth(trajectory, radius=30):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], np.ones(radius)/radius, mode='same')
    return smoothed_trajectory

if __name__ == "__main__":
    src_path = os.path.join(CUR_DIR, "data", "processed", "example_video_autozoom.mp4")
    dst_path = os.path.join(CUR_DIR, "data", "processed", "example_video_stabilized.mp4")
    # src_path = os.path.join(CUR_DIR, "data", "processed", "example_video_2_autozoom.mp4")
    # dst_path = os.path.join(CUR_DIR, "data", "processed", "example_video_2_stabilized.mp4")

    
    stabilize_video(src_path, dst_path)