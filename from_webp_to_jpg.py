# https://medium.com/@ajeet214/image-type-conversion-jpg-png-jpg-webp-png-webp-with-python-7d5df09394c9
from PIL import Image

im = Image.open("test.webp").convert('RGB')
im.save('test.jpg', format='jpeg')
