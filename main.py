from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import uuid
import torch
import base64
from seg1 import segment, model
from pattern import extract_attrs
from colors import dominant_colors

# segformer_b2_clothes(ATR 18클래스) 중 옷 클래스 번호
CLOTHING_GROUPS = {
    "upper": [4],           # Upper-clothes
    "lower": [5, 6],        # Skirt, Pants
    "full":  [7],           # Dress
    "all":   [4, 5, 6, 7],
}

# 부위별 물어볼 속성 (하의는 넥라인/소매 없음)
ATTRS_BY_CAT = {
    "upper": ["pattern", "neckline", "sleeve", "shoulder", "detail"],
    "lower": ["pattern", "detail"],
    "full":  ["pattern", "neckline", "sleeve", "shoulder", "detail"],
}

app = FastAPI(
    title="의류 분석 API",
    description="""
옷 사진을 입력하면 **누끼(배경 제거)** 와 **옷 정보(속성·색상)** 를 추출합니다.

- **누끼**: segformer_b2_clothes 기반. 상의/하의/전신 부위별 추출
- **정보추출**: FashionCLIP(디자인 속성) + K-means(대표색)

모든 엔드포인트는 이미지 파일을 `multipart/form-data` 의 `file` 필드로 업로드받습니다.
""",
    version="1.0.0",
)

# 누끼 결과 이미지를 저장·서빙할 폴더 (/static/파일명 으로 접근)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def make_cutout_png(image, pred, ids):
    """라벨맵(pred)에서 지정한 클래스(ids)에 해당하는 픽셀만 남긴 투명 PNG 바이트를 만든다."""
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for i in ids:
        mask |= (pred == i)
    mask_np = (mask.to(torch.uint8) * 255).cpu().numpy()
    mask_img = Image.fromarray(mask_np, mode="L")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/", tags=["기타"], summary="헬스체크")
def hello():
    """서버가 살아있는지 확인용."""
    return {"message": "HelloWorld!"}


@app.post("/segment", tags=["누끼"], summary="단일 부위 누끼 (PNG 이미지 반환)")
async def segment_endpoint(
    file: UploadFile = File(..., description="옷 사진 이미지 (jpg/png/webp)"),
    category: str = Query("all", description="누끼 부위 — upper(상의) / lower(하의) / full(원피스) / all(전체)"),
):
    """
    선택한 부위를 누끼(배경 제거)해서 **투명 PNG 이미지로 바로 반환**합니다.

    - **입력**: 이미지 파일 `file` + 쿼리 파라미터 `category`
    - **출력**: `image/png` — 응답 바디가 곧 PNG 바이트. 그대로 파일로 저장하면 이미지.
    - **요청 예**: `POST /segment?category=upper`
    - 부위가 사진에 없으면(예: 원피스에 lower 요청) 거의 빈 투명 이미지가 나올 수 있음.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred = segment(image)

    clothing_ids = CLOTHING_GROUPS.get(category, CLOTHING_GROUPS["all"])
    png = make_cutout_png(image, pred, clothing_ids)
    return Response(content=png, media_type="image/png")


@app.post("/segment2", tags=["누끼"], summary="상/하/전신 누끼 한 번에 (base64 JSON)")
async def segment_all_endpoint(
    file: UploadFile = File(..., description="옷 사진 이미지 (jpg/png/webp)"),
):
    """
    상의/하의/전신 3부위 누끼를 **한 번의 요청으로** 반환합니다 (모델 추론은 1회).

    - **입력**: 이미지 파일 `file`
    - **출력**: JSON — 각 부위 PNG를 base64 문자열로 인코딩
      ```json
      {"upper": "<base64 png>", "lower": "<base64 png>", "full": "<base64 png>"}
      ```
    - **사용법**: 각 값을 base64 디코드 → PNG 바이트 → 이미지로 저장/표시
    - 단일 파일이 필요하면 대신 `POST /segment?category=upper` 사용.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred = segment(image)

    result = {}
    for category in ["upper", "lower", "full"]:
        png = make_cutout_png(image, pred, CLOTHING_GROUPS[category])
        result[category] = base64.b64encode(png).decode("utf-8")
    return result


@app.post("/attrs", tags=["정보추출"], summary="옷 디자인 속성 추출 (무늬/넥라인/소매 등)")
async def attrs_endpoint(
    file: UploadFile = File(..., description="옷 사진 이미지 (누끼된 이미지 권장)"),
):
    """
    FashionCLIP 제로샷 분류로 옷의 디자인 속성을 추출합니다.

    - **입력**: 이미지 파일 `file` (누끼된 이미지를 넣으면 배경 영향이 줄어 더 정확)
    - **출력**: JSON `{속성: [라벨, 확신도]}`
      ```json
      {"pattern": ["floral", 0.44], "neckline": ["v neck", 0.45], "sleeve": ["sleeveless", 0.91]}
      ```
    - **속성 종류**: pattern(무늬) / neckline(넥라인) / sleeve(소매) / shoulder(숄더) / detail(디테일)
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return extract_attrs(image)


@app.post("/color", tags=["정보추출"], summary="대표색 추출")
async def color_endpoint(
    file: UploadFile = File(..., description="누끼된 이미지 (배경 투명 PNG 권장)"),
):
    """
    K-means로 대표색(팔레트)을 추출합니다.

    - **입력**: 이미지 파일 `file` (**누끼된 RGBA 이미지 권장** — 알파로 배경 픽셀 제외)
    - **출력**: JSON — 면적 비중 큰 순 대표색 목록
      ```json
      {"colors": [{"rgb": [210,180,140], "hex": "#d2b48c", "name": "tan", "ratio": 0.62}]}
      ```
    - `ratio` = 해당 색의 면적 비중(0~1), `name` = 가장 가까운 CSS 색이름.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return {"colors": dominant_colors(image)}


def _analyze(image, make_image_ref):
    """부위별 누끼+속성+색상 통합. make_image_ref(fname)이 image 필드 값(URL 또는 경로)을 만든다."""
    pred = segment(image)
    results = []
    for cat in ["upper", "lower", "full"]:
        # 부위 마스크
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for i in CLOTHING_GROUPS[cat]:
            mask |= (pred == i)
        if not bool(mask.any()):
            continue   # 사진에 이 부위 없음 → 제외

        # 누끼 이미지 → static 폴더에 저장
        mask_np = (mask.to(torch.uint8) * 255).cpu().numpy()
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(mask_np, mode="L"))
        fname = f"{uuid.uuid4().hex}_{cat}.png"     # 요청마다 고유 이름(충돌 방지)
        rgba.save(f"static/{fname}")

        # 속성 (부위별 · 흰 배경 합성해서 배경 영향 최소화)
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(white, rgba).convert("RGB")
        attrs = extract_attrs(comp, attr_names=ATTRS_BY_CAT[cat])
        meta_data = {a: label for a, (label, score) in attrs.items()}

        # 대표색 (1등 색이름)
        colors = dominant_colors(rgba)
        color = colors[0]["name"] if colors else None

        results.append({
            "image": make_image_ref(fname),
            "type": cat,
            "meta_data": meta_data,
            "color": color,
        })
    return results


@app.post("/analyze", tags=["통합"], summary="누끼+속성+색상 통합 (이미지 URL)")
async def analyze_endpoint(
    request: Request,
    file: UploadFile = File(..., description="옷 사진 이미지 (jpg/png/webp)"),
):
    """
    부위별(상의/하의/전신) 누끼 + 속성 + 대표색을 한 번에 반환. 누끼는 **URL**로.

    - **출력**: 부위마다 하나씩 리스트. 사진에 없는 부위는 제외.
      ```json
      [{"image": "http://<서버>/static/xxxx_upper.png", "type": "upper",
        "meta_data": {"pattern": "graphic print", "neckline": "v neck"}, "color": "black"}]
      ```
    - `image` = 누끼 PNG **URL**(GET 하면 이미지). 서버/네트워크가 다를 때 이 방식.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _analyze(image, lambda fname: f"{request.base_url}static/{fname}")


@app.post("/analyze_path", tags=["통합"], summary="누끼+속성+색상 통합 (이미지 파일 경로)")
async def analyze_path_endpoint(
    file: UploadFile = File(..., description="옷 사진 이미지 (jpg/png/webp)"),
):
    """
    `/analyze`와 동일하되, `image`를 URL이 아니라 **파일 경로**(`static/파일명`)로 반환.
    **같은 호스트에서 static 폴더를 볼륨으로 공유**하는 백엔드가 파일을 직접 읽을 때 사용.

    - **출력**:
      ```json
      [{"image": "static/xxxx_upper.png", "type": "upper", "meta_data": {...}, "color": "black"}]
      ```
    - `image` = static 폴더 기준 경로. 백엔드는 공유 볼륨의 static 위치와 합쳐 파일을 읽음.
    - **볼륨 공유 필요**: 실행 시 `-v <공유폴더>:/app/static` (백엔드도 같은 폴더 접근).
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _analyze(image, lambda fname: f"static/{fname}")
