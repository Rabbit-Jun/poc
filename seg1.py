import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from device_check import get_device

device =get_device()

model_name ="mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)

def segment(image):
    inputs = processor(images=image, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs =model(**inputs)

    logits = outputs.logits

    upsampled = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,

    )
    pred = upsampled.argmax(dim=1)[0]

    return pred

if __name__ == '__main__':
    image = Image.open("input/model2.jpg")
    pred = segment(image)

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
        rgba.save(f"output/{category}.png")
    print("saved")







