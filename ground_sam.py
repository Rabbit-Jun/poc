import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model
from device_check import get_device
from ground_detect import ground_detect
import numpy as np

device = get_device()
sam_name = 'facebook/sam2.1-hiera-tiny'
sam_processor = Sam2Processor.from_pretrained(sam_name)
sam_model = Sam2Model.from_pretrained(sam_name).to(device)

if __name__ == '__main__':
    image = Image.open("input/model5.webp").convert("RGB")

    text = "shirt. pants. dress. jacket. skirt. shoe."
    results = ground_detect(image, text)
    boxes = results["boxes"]
    labels = results["labels"]

    input_boxes = [[b.tolist() for b in boxes]]

    inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)
    masks = sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    for i in range(len(labels)):
        m =masks[i,0].cpu().numpy()
        mask_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")

        rgba = image.convert("RGBA")
        rgba.putalpha(mask_img)
        name = labels[i].replace(" ", "_")
        rgba.save(f"./output/sma2_{i}_{name}.png")
        print("saved", f"./output/sma2_{i}_{name}.png")
