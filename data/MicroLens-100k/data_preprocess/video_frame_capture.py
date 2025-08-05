import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

input_dir = r"/mnt/mydisk/videos"
output_dir = r"./data/MicroLens-100k/video_frames"
os.makedirs(output_dir, exist_ok=True)

log_file = "ffmpeg_fail_log.txt"

# 코덱별 GPU 디코더 매핑
GPU_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid"
}

def get_video_codec(video_path):
    """ffprobe로 비디오 코덱 감지"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    codec = result.stdout.strip().lower()
    return codec

def get_video_duration(video_path):
    """ffprobe로 비디오 길이(초) 가져오기"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout.strip())
    except Exception:
        duration = None
    return duration

def extract_10_frames(video_file):
    video_path = os.path.join(input_dir, video_file)
    video_id = os.path.splitext(video_file)[0]
    output_pattern = os.path.join(output_dir, f"{video_id}_%02d.jpg")

    # 1️⃣ 코덱 감지
    codec = get_video_codec(video_path)
    gpu_decoder = GPU_DECODERS.get(codec, None)

    # 2️⃣ 영상 길이 구하기
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        # 영상 길이 알 수 없으면 기본 fps 1로 시도
        fps_value = 1
    else:
        fps_value = 10 / duration  # 10프레임을 뽑기 위한 초당 프레임 수

    # 3️⃣ GPU 시도
    if gpu_decoder:
        cmd_gpu = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-c:v", gpu_decoder,
            "-i", video_path,
            "-vf", f"fps={fps_value}",
            "-vsync", "vfr",
            output_pattern
        ]
        gpu_result = subprocess.run(
            cmd_gpu,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        if gpu_result.returncode == 0:
            return f"[GPU OK] {video_id} ({codec})"
        else:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"[GPU FAIL] {video_file} ({codec}): {gpu_result.stderr[:300]}\n")

    # 4️⃣ CPU fallback
    cmd_cpu = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps_value}",
        "-vsync", "vfr",
        output_pattern
    ]
    cpu_result = subprocess.run(
        cmd_cpu,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        errors='replace'
    )

    if cpu_result.returncode == 0:
        return f"[CPU OK] {video_id} ({codec})"
    else:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"[CPU FAIL] {video_file} ({codec}): {cpu_result.stderr[:300]}\n")
        return f"[FAIL] {video_id} ({codec})"

if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    files = sorted(files)  # 파일명 오름차순 정렬

    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in tqdm(executor.map(extract_10_frames, files), total=len(files)):
            print(res)

    print("🎉 모든 영상에서 정확히 10프레임 추출 완료 (GPU 자동 선택 + CPU fallback)")
