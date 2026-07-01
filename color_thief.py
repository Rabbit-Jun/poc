from colorthief import ColorThief
import webcolors
from typing import List, Tuple



path = "./output/model1_upper.png"  # 분석할 이미지 경로

color_thief = ColorThief(path)
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
# build a color palette
palette = color_thief.get_palette(color_count=6)


def closest_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    RGB 값에서 가장 가까운 CSS3 색이름을 찾습니다.

    webcolors에는 '정확히 일치'하는 이름만 찾는 함수밖에 없으므로,
    모든 CSS3 색과의 유클리드 거리를 계산해 가장 가까운 이름을 직접 고릅니다.

    Args:
        rgb (Tuple[int, int, int]): RGB 값.

    Returns:
        str: 가장 가까운 CSS3 색이름.
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    best_name, best_dist = None, float('inf')
    for name in webcolors.names("css3"):
        cr, cg, cb = webcolors.name_to_rgb(name)
        dist = (cr - r) ** 2 + (cg - g) ** 2 + (cb - b) ** 2
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    RGB 값을 HEX 색상 코드로 변환합니다.

    Args:
        rgb (Tuple[int, int, int]): RGB 값.

    Returns:
        str: HEX 색상 코드.
    """
    r, g, b = rgb
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

# 대표색 1개 출력
print(
    f"Dominant Color: RGB={dominant_color}  "
    f"{rgb_to_hex(dominant_color)}  {closest_color_name(dominant_color)}"
)

# 팔레트(여러 대표색) 출력
print("Color Palette:")
for i, color in enumerate(palette):
    hex_code = rgb_to_hex(color)
    name = closest_color_name(color)
    print(f"  {i+1}. RGB={tuple(int(c) for c in color)}  {hex_code}  {name}")
