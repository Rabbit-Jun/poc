import torch
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

model_name ="mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)

image = Image.open("input/model2.jpg")

inputs = processor(images=image, return_tensors='pt').to(device)


with torch.no_grad():
    outputs =model(**inputs)

logits = outputs.logits

print("origin image size (W,H)", image.size)
print("logits shape :", logits.shape)


import torch.nn.functional as F

upsampled = F.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,

)
pred = upsampled.argmax(dim=1)[0]

print(model.config.id2label)
groups ={
    'upper-clothes': [4],
    'lower-clothes': [5,6],
    'full-clothes': [7]
}

for category,ids in groups.items():
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for i in ids:
        mask = mask | (pred == i)

    mask = mask.to(torch.uint8) * 255
    mask_np = mask.cpu().numpy()
    mask_img = Image.fromarray(mask_np, mode="L")

    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)
    rgba.save(f"{category}.png")
print("saved")