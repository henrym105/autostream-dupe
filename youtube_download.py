from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

def download_youtube_video(url, resolution, start_time=None, end_time=None):
    # Create a YouTube object
    yt = YouTube(url, on_progress_callback=on_progress)
    
    print(f"Title: {yt.title}")
    
    # Get the stream with the specified resolution
    stream = yt.streams.filter(res=resolution, progressive=True).first()
    output_video_name = yt.title + ".mp4"

    if stream:
        print(f"Downloading video in {resolution}...")
    else:
        print(f"No stream available in {resolution}. Trying highest available resolution...")
        stream = yt.streams.get_highest_resolution()
        print(f"Downloading video in {stream.resolution}...")
    
    # Add progress bar
    file_size = stream.filesize
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=output_video_name) as pbar:
        def progress_function(chunk, file_handle, bytes_remaining):
            pbar.update(file_size - bytes_remaining)
        
        video_path = stream.download(filename=output_video_name, on_progress_callback=progress_function)
    
    print("\nDownload completed successfully: ", video_path)

    # Trim the video if start_time and end_time are provided
    if start_time is not None or end_time is not None:
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=video_path)
        print(f"Video trimmed from {start_time} to {end_time} seconds.")



# Example usage
# video_url = input("Enter the YouTube video URL: ")
# desired_resolution = input("Enter the desired resolution (e.g., 720p, 1080p): ")
# start_time = int(input("Enter the start time in seconds: "))
# end_time = int(input("Enter the end time in seconds: "))
# video_url = "https://www.youtube.com/watch?v=EYDlrSTV_uI"

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=EYDlrSTV_uI"
    desired_resolution = "720p"
    start_time = 15  # Start at 30 seconds
    end_time = 25    # End at 60 seconds

    download_youtube_video(video_url, desired_resolution, start_time, end_time)
