import sys, time, torch
from PIL import Image
import pathlib

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["yolos", "dino", "segformer", "fashion-clip", "yolo_sam", "dino_sam", "rembg"]

def benchmark(fn,name,images, warmup=1, epoch=10):
    for _ in range(warmup):
        for img in images:
            fn(img)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(epoch):
        for img in images:
            fn(img)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    runs = epoch * len(images)
    ms = elapsed / runs* 1000
    vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"{name:25s} {ms:7.1f} ms/img peak VRAM {vram:.2f} GB ({runs} runs / {len(images)} images)")
    return ms, vram

if __name__ == "__main__":
    which = sys.argv[1]
    images = [Image.open(p).convert("RGB") for p in pathlib.Path("input").glob("*")]
    
    if which == 'all':
        import subprocess
        for model in MODELS:
            subprocess.run([sys.executable, __file__, model])
        sys.exit(0)


    if which == "yolos":
        from detect import detect_objects
        benchmark(lambda img: detect_objects(img), "yolos-fashionpedia",images=images)
        input("Press Enter to continue...")
    elif which == "dino":
        from ground_detect import ground_detect
        text = "shirt. pants. dress. jacket. skirt. shoe."
        benchmark(lambda img: ground_detect(img, text), "grounding-dino", images=images)
        input("Press Enter to continue...")

    elif which == "segformer":
        from seg1 import segment
        benchmark(lambda img: segment(img), "segformer", images=images)
        input("Press Enter to continue...")
    elif which == "fashion-clip": # 옷 정보 추출 모델
        from pattern import extract_attrs
        benchmark(lambda img: extract_attrs(img), "fashion-clip", images=images)
        input("Press Enter to continue...")
    elif which == "yolo_sam":
        from sam_cutout import yolo_sam
        benchmark(lambda img: yolo_sam(img), "yolo + sam", images=images)
        input("Press Enter to continue...")     
    elif which == "dino_sam":
        from ground_sam import ground_sam
        benchmark(lambda img: ground_sam(img, text= "shirt. pants. dress. jacket. skirt. shoe."), "grounding-dino + sam", images=images)
        input("Press Enter to continue...")
    elif which == "rembg":
        from seg_rembg import rembg_seg
        benchmark(lambda img: rembg_seg(img), "rembg", images=images)