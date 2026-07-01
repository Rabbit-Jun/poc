import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from device_check import get_device


device = get_device()
model_name ="IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)


def ground_detect(image,text, box_threshold=0.4, text_threshold=0.3):
    inputs = processor(images=image,text=text ,return_tensors='pt').to(device)
    print("inputs", inputs.keys())
    with torch.no_grad():
        outputs =model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        text_threshold=text_threshold,
        threshold=box_threshold,
        target_sizes=target_sizes,
    )[0]


    return results
 

if __name__ == "__main__":
    image = Image.open("input/model3.webp")
    text = "shirt. pants. dress. jacket. skirt. shoe."
    results = ground_detect(image, text)
    for score,label,box in zip(results["scores"], results["labels"], results["boxes"]):
        print(f"{label:20s} {score.item():.2f}  {[round(x,1) for x in box.tolist()]}")

    