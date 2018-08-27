# coords = [
#     [756, 411, 972, 563],
#     [789, 593, 1000, 766],
#     [578, 409, 749, 764],
#     [301, 403, 455, 553],
#     [96, 611, 524, 764],
#     [695, 889, 893, 1183],
#     [420, 883, 603, 1152],
#     [236, 991, 419, 1183],
#     [4, 958, 234, 1206],
#     [778, 1267, 1044, 1651],
#     [526, 1271, 708, 1624],
#     [65, 1354, 474, 1599]
# ]

coords = [
    [774, 405, 989, 528]
]

coords = [[int(coord[0]*0.5), int(coord[1]*0.5), int(coord[2]*0.5), int(coord[3]*0.5)] for coord in coords]

# classes = [b'avocado', b'orange', b'ketchup', b'onion', b'eggBox', b'joghurt', b'gerkin', b'pepper', b'pepper',
#            b'champagne', b'orangeJuice', b'tomato']

classes = [b'drone']

from PIL import Image, ImageDraw

# file_name = "/home/slapbot/my_side_projects/drone-detection/DataSets/Grocery/testImages/WIN_20160803_11_28_42_Pro.jpg"
file_name = "/home/slapbot/my_side_projects/drone-detection/DataSets/Drones/testImages/451.jpg"


img = Image.open(file_name).convert("RGBA")
img = img.resize((int(img.size[0]*0.5), int(img.size[1]*0.5)), Image.ANTIALIAS)
tmp = Image.new('RGBA', img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(tmp)

for index, coord in enumerate(coords):
    draw.rectangle(((coord[0], coord[1]), (coord[2], coord[3])), fill=(0, 0, 0, 127))
    draw.text((coord[0], coord[1]), classes[index])

img = Image.alpha_composite(img, tmp)

img.show()



