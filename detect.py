import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

model_name ="valentinafevu/yolos-fashionpedia"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)



image = Image.open("input/model3.webp")

inputs = processor(images=image, return_tensors='pt').to(device)


with torch.no_grad():
    outputs =model(**inputs)

print(outputs.keys())
print(model.config.id2label)



print("logits shape :", outputs.logits.shape)
print("pred_boxes shape :", outputs.pred_boxes.shape)

target_sizes = torch.tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs,
    threshold=0.5,
    target_sizes=target_sizes,
)[0]

import torchvision

keep = torchvision.ops.batched_nms(
    results["boxes"],
    results["scores"],
    results["labels"],
    iou_threshold=0.5
)

boxes = results["boxes"][keep]
scores = results["scores"][keep]
labels = results["labels"][keep]

for score, label, box in zip(scores, labels, boxes):
    name = model.config.id2label[label.item()]
    print(f"{name:30s} 확신도 {score.item():.2f} box {[round(x,1) for x in box.tolist()]}")

from PIL import ImageDraw

draw_img = image.convert("RGB").copy()
draw = ImageDraw.Draw(draw_img)

for score,label,box in zip(scores, labels, boxes):
    x1,y1,x2,y2 = box.tolist()
    draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
    name = model.config.id2label[label.item()]
    draw.text((x1,y1 -10), f"{name} {score.item():.2f}", fill= "red")

draw_img.save("detect_boxes.png")
print("detect_boxes.png saved")