from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import Response
from PIL import Image
import io
import torch
from seg1 import segment, model
from pattern import extract_attrs
from colors import dominant_colors
import base64

CLOTHING_GROUPS = {
    "upper": [4],           # Upper-clothes
    "lower": [5, 6],        # Skirt, Pants
    "full":  [7],           # Dress
    "all":   [4, 5, 6, 7],
}


app = FastAPI()

@app.get("/")
def hello():
    return {"message": "HelloWorld!"}


@app.post("/segment")
async def segment_endpoint(file: UploadFile = File(...), category: str = "all"):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pred = segment(image)

    clothing_ids = CLOTHING_GROUPS.get(category, CLOTHING_GROUPS["all"])
    mask = torch.zeros_like(pred, dtype=torch.bool)

    for i in clothing_ids:
        mask |= (pred == i)
    mask_np = (mask.to(torch.uint8) * 255).cpu().numpy()

    mask_image = Image.fromarray(mask_np, mode='L')
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_image)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

def make_cutout_png(image, pred, ids):          # 누끼 만드는 로직을 함수로 (중복 제거)
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

@app.post("/segment2")
async def segment_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred = segment(image)                         # 추론 1번만!

    result = {}
    for category in ["upper", "lower", "full"]:
        png = make_cutout_png(image, pred, CLOTHING_GROUPS[category])
        result[category] = base64.b64encode(png).decode("utf-8")   # 바이트→base64 문자열
    return result       # {"upper":"<base64>", "lower":"<base64>", "full":"<base64>"}


@app.post("/attrs")
async def attrs_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = extract_attrs(image)
    return result


@app.post("/color")
async def color_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return {"colors": dominant_colors(image)}
