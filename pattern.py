from transformers import pipeline
from device_check import get_device
from PIL import Image
import pathlib


device = get_device()
pipe = pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip", device=device)

groups ={
    "pattern": ["striped",  "floral", "polka dot", "geometric", "graphic print"],
    "neckline": ["crew neck", "v neck", "scoop neck",  "turtleneck","square neck"],
    "sleeve": ["short sleeve", "long sleeve", "sleeveless", "cap sleeve", "bell sleeve", "puff sleeve"],
    "shoulder" : ["off-shoulder", "cold shoulder", "one-shoulder", "drop shoulder", "regular shoulder"],
    "detail" : ["ruffles", "shirring", "pleats", "fringe", "embroidery", "no decoration"],
}

attrs_by_category = {
    "upper-clothes": ["pattern", "neckline", "sleeve", "shoulder", "detail"],
    "full-clothes": ["pattern","neckline","sleeve","shoulder" ,"detail",],
    "lower-clothes": ["pattern", "detail"],
}

def extract_attrs(image, attr_names=None):
    if attr_names is None:
        attr_names = groups.keys()
    results = {}
    for attr in attr_names:
        labels = groups[attr]
        res = pipe(image, candidate_labels=labels)
        top = res[0]
        results[attr] = (top['label'], top['score'])

    return results



if __name__ == '__main__':
    file_path = pathlib.Path('./output').glob('*')
    for path in list(file_path):
        image = Image.open(path)

        if path.name.startswith("upper"):
            attr_names = attrs_by_category['upper-clothes']
        elif path.name.startswith("lower"):
            attr_names = attrs_by_category['lower-clothes']
        else:
            attr_names = attrs_by_category['full-clothes']

        results = extract_attrs(image, attr_names)
        for attr, (label, score) in results.items():
            print(f"{attr:15s} {label:20s} {score:.4f}")
        print('-'*50)





