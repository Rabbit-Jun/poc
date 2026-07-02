from PIL import Image
from rembg import remove, new_session
import pathlib
import numpy as np


session = new_session("u2net_cloth_seg", providers=["CUDAExecutionProvider"])

def rembg_seg(image):
    return session.predict(image)


if __name__ == "__main__":
    img = Image.open("input/model2.jpg").convert("RGB")
    masks = rembg_seg(img)

    path = pathlib.Path('./input')
    output_path = pathlib.Path('./output')
    output_path.mkdir(exist_ok=True)
    categories = ["upper", "lower", "full"]
    
    for file in path.glob("*"):
        name = file.stem
        try:
            img= Image.open(file)
            origin = remove(img)

            masks =session.predict(origin)
            for category,mask in zip(categories, masks):
                mask_arr = np.array(mask)
                if (np.count_nonzero(mask_arr) / mask_arr.size) <= 0.01:
                    continue
                img_rgba = origin.convert("RGBA")      
                img_rgba.putalpha(mask)         
                img_rgba.save(output_path / f"rembg{name}_{category}.png")
        except Exception as e:
            print(f"Error  {e}")

