import os

def count_video_files(video_dir, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    count = 0
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            count += 1
    return count

# 사용 예시
video_dir = '/mnt/mydisk/videos'
video_count = count_video_files(video_dir)
print(f'비디오 개수: {video_count}')

#19738 개 