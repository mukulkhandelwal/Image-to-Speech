from PIL import Image
from tesseract import image_to_string

print(image_to_string(cv2.imread('abc.jpeg')))
#print (image_to_string(Image.open('abc.jpeg'), lang='eng'))