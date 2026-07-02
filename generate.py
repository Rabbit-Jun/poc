"""
각 누끼 모델의 상/하/전신 누끼 이미지를 생성 (품질 비교용).

실행 (4090에서, rembg의 CUDA 라이브러리 경로 필요):
    export LD_LIBRARY_PATH="$(find $PWD/.venv -type d -path '*/nvidia/*/lib' | tr '\n' ':')$LD_LIBRARY_PATH"
    python generate.py

결과: output/compare/<모델>/<이미지>_<카테고리>.png
      예) output/compare/segformer/model3_upper.png
"""
import torch
import numpy as np
import pathlib
from PIL import Image

from seg1 import segment              # segformer
from sam_cutout import yolo_sam       # yolos + SAM2 (labels = 클래스번호)
from ground_sam import ground_sam     # GroundingDINO + SAM2 (labels = 텍스트)

# rembg는 onnxruntime → 미설치 환경이면 자동 스킵
try:
    from seg_rembg import rembg_seg
    HAS_REMBG = True
except Exception as e:
    print(f"[rembg 스킵] {e}")
    HAS_REMBG = False


def apply_mask(image, mask_bool):
    """bool 마스크(numpy [H,W]) → 원본에 알파로 씌운 누끼 RGBA."""
    mask_np = (mask_bool.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_np, mode="L")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)
    return rgba


# ---- 모델별 카테고리 매핑 ----
SEGFORMER_GROUPS = {"upper": [4], "lower": [5, 6], "full": [7]}   # ATR 18클래스

YOLOS_GROUPS = {                                                  # fashionpedia 46클래스 id
    "upper": {0, 1, 2, 3, 4, 5, 9},   # shirt/top/sweater/cardigan/jacket/vest/coat
    "lower": {6, 7, 8},               # pants/shorts/skirt
    "full":  {10, 11},                # dress/jumpsuit
}


def dino_category(label):                                        # dino 텍스트 라벨 → 카테고리
    l = str(label).lower()
    if any(w in l for w in ["pants", "skirt", "shorts", "trouser"]):
        return "lower"
    if any(w in l for w in ["dress", "jumpsuit"]):
        return "full"
    if any(w in l for w in ["shirt", "jacket", "top", "sweater", "cardigan", "vest", "coat", "blouse"]):
        return "upper"
    return None   # shoe 등 옷 아님 → 스킵


def cutout_segformer(image):
    """segformer: 라벨맵 → 카테고리별 클래스 마스크."""
    pred = segment(image)
    result = {}
    for cat, ids in SEGFORMER_GROUPS.items():
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for i in ids:
            mask |= (pred == i)
        if bool(mask.any()):
            result[cat] = apply_mask(image, mask.cpu().numpy())
    return result


def _group_sam_masks(image, masks, cats):
    """cats: 각 마스크의 카테고리. 카테고리별로 마스크 union → 누끼 dict."""
    result = {}
    for cat in ["upper", "lower", "full"]:
        idxs = [i for i, c in enumerate(cats) if c == cat]
        if not idxs:
            continue
        union = masks[idxs, 0].any(dim=0).cpu().numpy()
        result[cat] = apply_mask(image, union)
    return result


def cutout_yolo_sam(image):
    masks, labels = yolo_sam(image)
    if len(masks) == 0:
        return {}
    cats = []
    for i in range(len(labels)):
        cid = labels[i].item()
        cat = next((c for c, ids in YOLOS_GROUPS.items() if cid in ids), None)
        cats.append(cat)
    return _group_sam_masks(image, masks, cats)


def cutout_ground_sam(image):
    masks, labels = ground_sam(image)
    if len(masks) == 0:
        return {}
    cats = [dino_category(labels[i]) for i in range(len(labels))]
    return _group_sam_masks(image, masks, cats)


def cutout_rembg(image):
    """rembg: 상/하/전신 3마스크가 순서대로 나옴."""
    masks = rembg_seg(image)
    result = {}
    for cat, m in zip(["upper", "lower", "full"], masks):
        arr = np.array(m) > 127
        if arr.any():
            result[cat] = apply_mask(image, arr)
    return result


MODELS = {
    "segformer":  cutout_segformer,
    "yolo_sam":   cutout_yolo_sam,
    "ground_sam": cutout_ground_sam,
}
if HAS_REMBG:
    MODELS["rembg"] = cutout_rembg


if __name__ == "__main__":
    out_root = pathlib.Path("output/compare")
    for name in MODELS:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    for img_path in sorted(pathlib.Path("input").glob("*")):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"open error {img_path.name}: {e}")
            continue

        stem = img_path.stem
        for name, fn in MODELS.items():
            try:
                cutouts = fn(image)   # {category: rgba}
                if not cutouts:
                    print(f"skip  {name}/{stem} (검출/마스크 없음)")
                    continue
                for cat, rgba in cutouts.items():
                    save_path = out_root / name / f"{stem}_{cat}.png"
                    rgba.save(save_path)
                    print(f"saved {save_path}")
            except Exception as e:
                print(f"error {name}/{stem}: {e}")

    print("\n완료. 결과: output/compare/<모델>/<이미지>_<카테고리>.png")
