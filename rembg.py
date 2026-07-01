from PIL import Image
from rembg import remove, new_session
from rembg.sessions import BaseSession
import pathlib
import numpy as np


session = new_session("u2net_cloth_seg")
path = pathlib.Path('./input')
files = path.glob('*')
output_path = pathlib.Path('./output')
output_path.mkdir(exist_ok=True)

def process_one(file_path:str,name:str, session: BaseSession) -> None:

    try:
        img= Image.open(file_path)
        origin = remove(img)
        categories = ["upper", "lower", "full"]

        masks =session.predict(origin)
  
        for category,mask in zip(categories, masks):
            mask_arr = np.array(mask)
            if (np.count_nonzero(mask_arr) / mask_arr.size) <= 0.01:
                continue
            img_rgba = origin.convert("RGBA")      
            img_rgba.putalpha(mask)         
            img_rgba.save(output_path / f"{name}_{category}.png")
    except Exception as e:
        print(f"Error mask processing file name:{name}: {e}")



for file in files:
    name = file.stem
    process_one(file,name, session)

