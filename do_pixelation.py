from PIL import Image

img = Image.open("C://Users//Lenovo//Desktop//Intel_Unnati 2024//Data//Tower.jpg")

imgSmall = img.resize((16,16), resample=Image.Resampling.BILINEAR)

result = imgSmall.resize(img.size, Image.Resampling.NEAREST)

result.save('C://Users//Lenovo//Desktop//Intel_Unnati 2024//Data//Tower1.jpg')