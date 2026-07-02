import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model
from device_check import get_device
from detect import detect_objects, model as yolo
import numpy as np

device = get_device()
sam_name = 'facebook/sam2.1-hiera-tiny'
sam_processor = Sam2Processor.from_pretrained(sam_name)
sam_model = Sam2Model.from_pretrained(sam_name).to(device)

def yolo_sam(image):
    boxes, scores, labels = detect_objects(image)
    input_boxes = [[b.tolist() for b in boxes]]
    inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)
    masks = sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    return masks, labels

  


if __name__ == '__main__':
    image = Image.open("input/model5.webp").convert("RGB")
    for i in range(len(labels)):
        m =masks[i,0].cpu().numpy()
        mask_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")

        rgba = image.convert("RGBA")
        rgba.putalpha(mask_img)
        name = yolo.config.id2label[labels[i].item()].split(",")[0].replace(" ", "_")
        rgba.save(f"./output/sma_{i}_{name}.png")
        print("saved", name)

