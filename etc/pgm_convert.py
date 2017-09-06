# pgm_convert.py
from PIL import Image
import matplotlib.pyplot as plt
im = Image.open("img/hero.png")
# im = im.convert('RGB')
im.save('img/hero_1.pgm')

i1 = plt.imread('img/hero_1.pgm')
plt.imshow(i1)
plt.show()