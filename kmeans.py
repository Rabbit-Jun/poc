from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from typing import List, Tuple
import webcolors


def get_valid_pixels(image: Image.Image, quality: int = 1) -> List[Tuple[int, int, int]]:
    """
    이미지에서 유효한 픽셀(RGB)을 추출합니다.

    Args:
        image (Image.Image): PIL 이미지 객체.
        quality (int): 처리 품질(샘플링 간격). 값이 클수록 더 적은 픽셀을 샘플링합니다.

    Returns:
        List[Tuple[int, int, int]]: 유효한 픽셀의 RGB 값 리스트.
    """
    width, height = image.size
    pixels = image.getdata()
    pixel_count = width * height
    valid_pixels = []

    # 품질(quality)에 따라 샘플링하며 유효한 픽셀 필터링
    for i in range(0, pixel_count, quality):
        r, g, b, a = pixels[i]
        # 알파 값이 125 이상이고 흰색이 아닌 경우만 유효 픽셀로 간주
        if a >= 125:
            if not (r > 250 and g > 250 and b > 250):  # 완전 흰색 제외
                valid_pixels.append((r, g, b))
    return valid_pixels


def get_unique_pixels(valid_pixels: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    유효한 픽셀에서 중복을 제거하고 고유한 픽셀만 반환합니다.

    Args:
        valid_pixels (List[Tuple[int, int, int]]): 유효한 픽셀의 RGB 값 리스트.

    Returns:
        np.ndarray: 고유한 픽셀의 RGB 값 배열.
    """
    return np.unique(valid_pixels, axis=0)


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


def plot_valid_pixels(valid_pixels: List[Tuple[int, int, int]]) -> None:
    """
    유효한 픽셀을 3D RGB 공간에 시각화합니다.

    Args:
        valid_pixels (List[Tuple[int, int, int]]): 유효한 픽셀의 RGB 값 리스트.

    Returns:
        None
    """
    unique_pixels = get_unique_pixels(valid_pixels)

    # 고유 픽셀을 HEX 색상으로 변환
    hhex = list(map(rgb_to_hex, unique_pixels))

    # 3D 플롯 생성
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = unique_pixels[:, 0]  # R 채널
    y = unique_pixels[:, 1]  # G 채널
    z = unique_pixels[:, 2]  # B 채널

    ax.scatter(x, y, z, color=hhex)

    ax.set_xlabel('R (0~255)')
    ax.set_ylabel('G (0~255)')
    ax.set_zlabel('B (0~255)')

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


def extract_dominant_colors(
    valid_pixels: List[Tuple[int, int, int]], k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means 클러스터링을 사용하여 주요 색상과 면적 비중을 추출합니다.

    중복을 제거하지 않고 '전체 픽셀'로 군집하므로, 넓은 면적의 색일수록
    더 많은 픽셀이 모여 실제 '대표색(dominant)'에 가깝게 잡힙니다.

    Args:
        valid_pixels (List[Tuple[int, int, int]]): 유효한 픽셀의 RGB 값 리스트.
        k (int): 추출할 주요 색상의 개수.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 주요 색상의 RGB 값 배열 (비중 내림차순)
            - 각 색상의 면적 비중(0~1) 배열 (비중 내림차순)
    """

    pixels = np.array(valid_pixels)

    # K-Means 클러스터링 수행 (전체 픽셀 → 면적 반영)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixels)

    # 각 군집에 속한 픽셀 수 → 면적 비중 계산
    counts = np.bincount(labels, minlength=k)
    ratios = counts / counts.sum()

    centers = np.array(kmeans.cluster_centers_, dtype='uint8')

    # 비중 내림차순으로 정렬 (가장 넓은 색이 맨 위)
    order = np.argsort(ratios)[::-1]
    return centers[order], ratios[order]


def plot_dominant_colors(dominant_colors: np.ndarray, ratios: np.ndarray) -> None:
    """
    주요 색상을 비중·색이름과 함께 콘솔에 출력합니다.

    Args:
        dominant_colors (np.ndarray): 주요 색상의 RGB 값 배열.
        ratios (np.ndarray): 각 색상의 면적 비중(0~1) 배열.

    Returns:
        None
    """

    # 콘솔 출력: RGB / HEX / 색이름 / 비중
    print("Dominant Colors:")
    for i, (color, ratio) in enumerate(zip(dominant_colors, ratios)):
        hex_code = rgb_to_hex(tuple(color))
        name = closest_color_name(tuple(color))
        print(
            f"  {i+1}. RGB={tuple(int(c) for c in color)}  "
            f"{hex_code}  {name:<20}  {ratio*100:5.1f}%"
        )




if __name__ == "__main__":
    path = "./output/model1_upper.png"  # 분석할 이미지 경로
    image = Image.open(path)
    image = image.convert("RGBA")

    valid_pixels = get_valid_pixels(image)

    plot_valid_pixels(valid_pixels)

    dominant_colors, ratios = extract_dominant_colors(valid_pixels)
    plot_dominant_colors(dominant_colors, ratios)