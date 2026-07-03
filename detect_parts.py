"""
yolos-fashionpedia로 옷의 구조/장식 부위(넥라인·소매·주머니·러플·지퍼 등)를 검출.
FashionCLIP(속성 "분류")과 다른 방식 — 부위의 존재+위치를 "검출"(박스).

실행 (torch만 → Mac/4090 어디서든):
    python detect_parts.py
결과: 콘솔 출력 + output/parts.json
"""
import json
import pathlib
from PIL import Image
from detect import detect_objects, model as yolo

# yolos-fashionpedia 46클래스 중 "구조/장식 부위" (27~45)
# hood, collar, lapel, epaulette, sleeve, pocket, neckline, buckle, zipper,
# applique, bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel
PART_IDS = set(range(27, 46))


def detect_parts(image, threshold=0.4):
    """이미지에서 구조/장식 부위만 골라 반환: [{part, score, box}, ...] (확신도 높은 순)."""
    boxes, scores, labels = detect_objects(image, threshold=threshold)
    parts = []
    for i in range(len(labels)):
        cid = labels[i].item()
        if cid in PART_IDS:
            parts.append({
                "part": yolo.config.id2label[cid],
                "score": round(scores[i].item(), 3),
                "box": [round(x, 1) for x in boxes[i].tolist()],
            })
    parts.sort(key=lambda p: p["score"], reverse=True)
    return parts


if __name__ == "__main__":
    pathlib.Path("output").mkdir(exist_ok=True)
    result = {}

    for p in sorted(pathlib.Path("input").glob("*")):
        try:
            image = Image.open(p).convert("RGB")
            parts = detect_parts(image)
            result[p.stem] = parts

            print(f"\n=== {p.stem} ===")
            if not parts:
                print("  (검출된 부위 없음)")
            for item in parts:
                print(f"  {item['part']:20s} {item['score']:.2f}")
        except Exception as e:
            print(f"error {p.stem}: {e}")

    with open("output/parts.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nsaved output/parts.json")
