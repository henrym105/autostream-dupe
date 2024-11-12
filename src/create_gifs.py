import moviepy.editor as mp
import os
from constants import CUR_DIR

def convert_mp4_to_gif(input_file, output_file, width, start_time, end_time, fps=10):
    # Load the video file
    clip = mp.VideoFileClip(input_file)
    
    # Trim the video if end_time is not -1
    if end_time != -1:
        clip = clip.subclip(start_time, end_time)
    else:
        clip = clip.subclip(start_time)
    
    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = clip.size[1] / clip.size[0]
    height = int(width * aspect_ratio)
    
    # Resize the video
    clip = clip.resize(newsize=(width, height))
    
    # Make the gif loop
    # clip = clip.loop(duration=(end_time - start_time) * 2 if end_time != -1 else clip.duration * 2)
    
    # Write the result to a gif file
    clip.write_gif(output_file, fps=fps)

# Example usage:
if __name__ == "__main__":
    src_path = os.path.join(CUR_DIR, "data", "processed", "example_video_2_autozoom.mp4")
    dst_path = os.path.join(CUR_DIR, "data", "processed", "example_video_autozoom.gif")

    convert_mp4_to_gif(src_path, dst_path, 500, 10, -1, fps=10)
