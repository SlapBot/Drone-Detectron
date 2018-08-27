import time
from PIL import Image, ImageDraw


def visualize(visualizer):
    img = Image.open(visualizer['imgPath']).convert("RGBA")
    tmp = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)

    for index, coord in enumerate(visualizer['gtBoxes']):
        draw.rectangle(((coord[0], coord[1]), (coord[2], coord[3])), fill=(0, 0, 0, 127))
        draw.text((coord[0], coord[1]), visualizer['gtLabels'][index])

    img = Image.alpha_composite(img, tmp)
    img.show()


def visualize_multiple(visualizers, count=1):
    for index, visualizer in enumerate(visualizers):
        if index < count:
            visualize(visualizer)
            time.sleep(1)
