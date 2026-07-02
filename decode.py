import json, base64, os

os.makedirs("output", exist_ok=True)          # 폴더 보장
d = json.load(open("result.json"))
for category, b64_string in d.items():
    with open(f"output/{category}.png", "wb") as f:   # / 없이!
        f.write(base64.b64decode(b64_string))
    print("saved", f"output/{category}.png")