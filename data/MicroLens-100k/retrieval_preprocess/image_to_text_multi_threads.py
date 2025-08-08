import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def loading_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    return processor, model


def convert_image_to_text(processor, model, image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text


def process_row(item_id, path, processor, model):
    current_text_list = []
    for j in range(1, 11):
        image_path = os.path.join(path, f"{item_id}_{j:02d}.jpg") 
        if not os.path.exists(image_path):
            print(f"❌ 이미지 없음: {image_path}")
            current_text_list.append("")
            continue
        try:
            text = convert_image_to_text(processor, model, image_path)
        except Exception as e:
            print(f"⚠️ 변환 실패 ({image_path}): {e}")
            text = ""
        current_text_list.append(text)
    return item_id, current_text_list


def extract_available_item_ids(image_dir):
    item_ids = set()
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            parts = filename.split("_")
            if len(parts) == 2 and parts[1].endswith(".jpg"):
                try:
                    item_id = int(parts[0])
                    item_ids.add(item_id)
                except ValueError:
                    continue
    return sorted(list(item_ids))


def main():
    image_dir = "./data/MicroLens-100k/video_frames"
    output_csv_path = "./blip_caption_results.csv"

    # 💡 실제 존재하는 item_id 자동 추출
    item_ids = extract_available_item_ids(image_dir)
    print(f"🎯 총 {len(item_ids)}개의 item_id 발견됨")

    processor, model = loading_model()

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_row, item_id, image_dir, processor, model) for item_id in item_ids]

        for future in tqdm(as_completed(futures), total=len(futures)):
            item_id, captions = future.result()
            results.append({
                "item_id": item_id,
                **{f"caption_{i}": captions[i] for i in range(10)}
            })

    df = pd.DataFrame(results)
    df.sort_values("item_id", inplace=True)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ CSV 저장 완료: {output_csv_path}")


if __name__ == "__main__":
    main()
