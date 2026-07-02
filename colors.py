import numpy as np
from sklearn.cluster import KMeans
import webcolors


def _closest_name(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    best, bestd = None, float("inf")
    for name in webcolors.names('css3'):
        cr, cg, cb = webcolors.name_to_rgb(name)
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < bestd:
            bestd = d
            best = name
    return best

def dominant_colors(image, k=5, quality=1):
    image = image.convert("RGBA")
    pixels = list(image.getdata())
    valid = [(r, g, b) for (r, g, b, a) in pixels[::quality]
             if a >= 125 and not (r > 250 and g > 250 and b > 250)]
    if not valid:
        return []
    arr = np.array(valid)
    km = KMeans(n_clusters=min(k,len(arr)), random_state=42, n_init='auto')
    labels = km.fit_predict(arr)
    ratios = np.bincount(labels) / len(labels)
    centers = km.cluster_centers_.astype(int)
    order = np.argsort(ratios)[::-1]
    return [{
        "rgb": [int(centers[i][0]), int(centers[i][1]), int(centers[i][2])],
        "hex": "#%02x%02x%02x" % tuple(int(x) for x in centers[i]),
        "name": _closest_name(centers[i]),
        "ratio": round(float(ratios[i]), 3),
    } for i in order]