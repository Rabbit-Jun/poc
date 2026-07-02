"""
각 누끼 모델의 결과 이미지를 몰아서 생성 (품질 비교용).

실행 (4090에서, rembg의 CUDA 라이브러리 경로 필요):
    export LD_LIBRARY_PATH="$(find $PWD/.venv -type d -path '*/nvidia/*/lib' | tr '\n' ':')$LD_LIBRARY_PATH"
    python generate.py

결과: output/compare/<모델명>/<이미지명>.png  (모델별 폴더로 정리)
      → Google Drive 업로드 → Confluence 링크
"""
import torch
import numpy as np
import pathlib
from PIL import Image

from seg1 import segment              # segformer_b2_clothes
from sam_cutout import yolo_sam       # yolos + SAM2
from ground_sam import ground_sam     # GroundingDINO + SAM2
from seg_rembg import rembg_seg       # rembg (u2net_cloth_seg)


def apply_mask(image, mask_bool):
    """bool 마스크(numpy [H,W]) → 원본에 알파로 씌운 누끼 RGBA 이미지."""
    mask_np = (mask_bool.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_np, mode="L")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)
    return rgba


def cutout_segformer(image):
    """segformer: 라벨맵 → 옷 클래스(상의/치마/바지/원피스) 합쳐 누끼."""
    pred = segment(image)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for i in [4, 5, 6, 7]:
        mask |= (pred == i)
    return apply_mask(image, mask.cpu().numpy())


def _sam_union(masks):
    """yolo_sam/ground_sam 공통: 여러 인스턴스 마스크를 하나로 합침([N,1,H,W]→[H,W])."""
    if len(masks) == 0:
        return None
    return masks[:, 0].any(dim=0).cpu().numpy()


def cutout_yolo_sam(image):
    masks, labels = yolo_sam(image)
    u = _sam_union(masks)
    return apply_mask(image, u) if u is not None else None


def cutout_ground_sam(image):
    masks, labels = ground_sam(image)
    u = _sam_union(masks)
    return apply_mask(image, u) if u is not None else None


def cutout_rembg(image):
    """rembg: 상/하/전신 3개 마스크를 합쳐 누끼."""
    masks = rembg_seg(image)
    combined = None
    for m in masks:
        arr = np.array(m) > 127
        combined = arr if combined is None else (combined | arr)
    if combined is None:
        return None
    return apply_mask(image, combined)


# 모델명 → 누끼 함수 (모델 추가는 여기 한 줄)
MODELS = {
    "segformer":  cutout_segformer,
    "yolo_sam":   cutout_yolo_sam,
    "ground_sam": cutout_ground_sam,
    "rembg":      cutout_rembg,
}


if __name__ == "__main__":
    out_root = pathlib.Path("output/compare")
    for name in MODELS:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    input_dir = pathlib.Path("input")
    for img_path in sorted(input_dir.glob("*")):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"open error {img_path.name}: {e}")
            continue

        stem = img_path.stem
        for name, fn in MODELS.items():
            try:
                result = fn(image)
                if result is not None:
                    save_path = out_root / name / f"{stem}.png"
                    result.save(save_path)
                    print(f"saved {save_path}")
                else:
                    print(f"skip  {name}/{stem} (검출/마스크 없음)")
            except Exception as e:
                print(f"error {name}/{stem}: {e}")

    print("\n완료. 결과: output/compare/<모델명>/<이미지명>.png")
