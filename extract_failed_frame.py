# import os

# frame_dir = "/home/changjin/바탕화면/sosp2/MMRA/data/MicroLens-100k/video_frames"

# video_ids = set()

# for filename in os.listdir(frame_dir):
#     if filename.endswith(".jpg"):
#         video_id = filename.split("_")[0]
#         video_ids.add(video_id)

# print(f"✅ 고유한 영상 ID 개수: {len(video_ids)}")









#######################################
# 실패 영상들 나머지 프레임 삭제 코드 #
#######################################

# import os
# from tqdm import tqdm

# frame_folder = 'data/MicroLens-100k/video_frames'
# problem_ids = ['10006', '13033', '14632']

# for pid in problem_ids:
#     for i in range(1, 10):  # 10개 프레임 예상
#         frame_path = os.path.join(frame_folder, f"{pid}_{i:02d}.jpg")
#         if os.path.exists(frame_path):
#             os.remove(frame_path)
#             print(f"삭제됨: {frame_path}")
#         else:
#             # 없는 파일은 넘어가기
#             continue
# print("문제 영상 프레임 삭제 완료!")






from tqdm import tqdm
import cv2
import os
import numpy as np

def extract_frames(input_video, input_video_id, k):
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"❌ 영상 {input_video_id}는 프레임 없음 (total_frames=0)")
        return

    output_folder = 'data/MicroLens-100k/video_frames'
    
    os.makedirs(output_folder, exist_ok=True)

    frame_indices = np.linspace(0, total_frames - 1, k).astype(int)

    saved_frames = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 영상 {input_video_id}: 프레임 {frame_index} 읽기 실패")
            continue

        # 저장 파일명 01 ~ 10
        output_path = os.path.join(output_folder, f"{input_video_id}_{saved_frames + 1:02d}.jpg")
        print(f"{saved_frames + 1:02d}")
        cv2.imwrite(output_path, frame)
        saved_frames += 1

    cap.release()

    if saved_frames < k:
        print(f"⚠️ 영상 {input_video_id}: {saved_frames}/{k}개 프레임만 저장됨")



if __name__ == '__main__':
    path = '/home/changjin/바탕화면/dd'  # 실제 영상 폴더 경로
    files = os.listdir(path)
    k = 10

    problem_ids = ['10006', '13033', '14632']

    for file in tqdm(files):
        input_video_id = file[:-4]
        if input_video_id in problem_ids:
            input_video_path = os.path.join(path, file)
            extract_frames(input_video_path, input_video_id, k)
