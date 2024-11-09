from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm
import os

def download_youtube_video(url, resolution, start_time=None, end_time=None):
    # Create a YouTube object
    yt = YouTube(url, on_progress_callback=on_progress)
    
    print(f"Title: {yt.title}")
    
    # Get the stream with the specified resolution
    stream = yt.streams.filter(res=resolution, progressive=True).first()
    data_dir = os.path.join(os.getcwd(), "data", "raw")
    output_video_name = os.path.join(data_dir, f"{yt.title}.mp4")

    if stream:
        print(f"Downloading video in {resolution}...")
    else:
        print(f"No stream available in {resolution}. Trying highest available resolution...")
        stream = yt.streams.get_highest_resolution()
        print(f"Downloading video in {stream.resolution}...")
    
    # Download the video
    video_path = stream.download(filename=output_video_name)    
    print("\nDownload completed successfully: ", video_path)

    # Trim the video if start_time and end_time are provided
    if start_time is not None or end_time is not None:
        trimmed_video_path = os.path.join(data_dir, f"trimmed_video_path_{yt.title}.mp4")

        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=trimmed_video_path)
        print(f"Video trimmed from {start_time} to {end_time} seconds.")
        print(f"Trimmed video saved as: {trimmed_video_path}")
        
        # delete the original video
        if os.path.exists(video_path): os.remove(video_path)


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?app=desktop&v=Yjni2VguRLU"
    desired_resolution = "480p"
    start_time = 15  # Start at 30 seconds
    end_time = 25    # End at 60 seconds

    download_youtube_video(video_url, desired_resolution, start_time, end_time)
