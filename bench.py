import sys, time, torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark(fn,name, warmup=3, epoch=20):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(epoch):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed / epoch * 1000
    vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"{name:25s} {ms:7.1f} ms/img peak VRAM {vram:.2f} GB")
    return ms, vram

if __name__ == "__main__":
    which = sys.argv[1]
    image = Image.open("input/model3.webp").convert("RGB")
    
    if which == "yolos":
        from detect import detect_objects
        benchmark(lambda: detect_objects(image), "yolos-fashionpedia")
    elif which == "dino":
        from ground_detect import ground_detect
        text = "shirt. pants. dress. jacket. skirt. shoe."
        benchmark(lambda: ground_detect(image, text), "grounding-dino")
    elif which == "segformer":
        from seg1 import segment
        benchmark(lambda: segment(image), "segformer")
    elif which == "fashion-clip": # 옷 정보 추출 모델
        from pattern import extract_attrs
        benchmark(lambda: extract_attrs(image), "fashion-clip")
    elif which == "yolo_sam":
        from sam_cutout import yolo_sam
        benchmark(lambda: yolo_sam(image), "yolo + sam")
    elif which == "dino_sam":
        from ground_sam import ground_sam
        benchmark(lambda: ground_sam(image, text= "shirt. pants. dress. jacket. skirt. shoe."), "grounding-dino + sam")