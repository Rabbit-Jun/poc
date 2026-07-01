import torch
from PIL import Image,ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torchvision
from device_check import get_device


device = get_device()
model_name ="valentinafevu/yolos-fashionpedia"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)



def detect_objects(image, threshold=0.5, iou=0.5):
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs =model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    keep = torchvision.ops.batched_nms(
        results["boxes"],
        results["scores"],
        results["labels"],
        iou_threshold=iou
    )

    return results["boxes"][keep], results["scores"][keep], results["labels"][keep]  
 




def draw_boxes(image, boxes, scores, labels):
    draw_img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_img)

    for score,label,box in zip(scores, labels, boxes):
        x1,y1,x2,y2 = box.tolist()
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        name = model.config.id2label[label.item()]
        draw.text((x1,y1 -10), f"{name} {score.item():.2f}", fill= "red")

    return draw_img

if __name__ == "__main__":
    image = Image.open("input/model3.webp")
    boxes, scores, labels = detect_objects(image, threshold=0.5, iou=0.5)

    for score,label,box in zip(scores, labels, boxes):
                print(f"{model.config.id2label[label.item()]:30s} {score.item():.2f}  {[round(x,1) for x in box.tolist()]}")

    draw_boxes(image, boxes, scores, labels).save("detected_image.png")
    