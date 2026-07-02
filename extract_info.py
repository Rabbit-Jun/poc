"""
각 입력 이미지의 옷 정보(속성 + 대표색)를 부위(upper/lower/full)별로 추출 → JSON.
- segformer로 부위별 누끼를 만들고, 그 누끼 이미지로 속성·색상 추출 (배경 영향 제거)
- 부위마다 물어보는 속성이 다름: 하의는 neckline/sleeve 안 물어봄

실행 (Mac MPS / 4090 CUDA 어디서든, rembg 불필요):
    python extract_info.py
결과: output/info.json  구조: {이미지: {부위: {attrs, colors}}}
"""
import json
import pathlib
import torch
import numpy as np
from PIL import Image

from seg1 import segment
from pattern import extract_attrs
from colors import dominant_colors

# 부위 → segformer 클래스 번호
CAT_GROUPS = {"upper": [4], "lower": [5, 6], "full": [7]}

# 부위 → 물어볼 속성 그룹 (하의엔 넥라인/소매 없음)
ATTRS_BY_CAT = {
    "upper": ["pattern", "neckline", "sleeve", "shoulder", "detail"],
    "lower": ["pattern", "detail"],
    "full":  ["pattern", "neckline", "sleeve", "shoulder", "detail"],
}


def category_cutout(image, pred, ids):
    """해당 부위 클래스만 남긴 누끼 RGBA. 부위가 없으면 None."""
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for i in ids:
        mask |= (pred == i)
    if not bool(mask.any()):
        return None
    mask_np = (mask.to(torch.uint8) * 255).cpu().numpy()
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_np, mode="L"))
    return rgba


def composite_white(rgba):
    """누끼(투명 배경)를 흰 배경 위에 합성 → RGB (FashionCLIP 입력용, 배경 영향 최소화)."""
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


if __name__ == "__main__":
    pathlib.Path("output").mkdir(exist_ok=True)
    result = {}

    for p in sorted(pathlib.Path("input").glob("*")):
        try:
            image = Image.open(p).convert("RGB")
            pred = segment(image)

            per_cat = {}
            for cat, ids in CAT_GROUPS.items():
                cut = category_cutout(image, pred, ids)
                if cut is None:
                    continue   # 이 부위 없음 (예: 원피스엔 lower 없음)
                attrs = extract_attrs(composite_white(cut), attr_names=ATTRS_BY_CAT[cat])
                colors = dominant_colors(cut)   # 누끼(알파)로 옷 색만
                per_cat[cat] = {"attrs": attrs, "colors": colors}

            result[p.stem] = per_cat
            print(p.stem, "→", list(per_cat.keys()))
        except Exception as e:
            print(f"error {p.stem}: {e}")

    with open("output/info.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nsaved output/info.json")
