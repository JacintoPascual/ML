import requests
from bs4 import BeautifulSoup
import operator

answer = lambda x: x*7
print(answer(5))

stocks = {
    'GOOD': 520.54,
    'FB': 76.45,
    'YHOO': 39.78,
    'AAPL': 99.76
}

print(min((zip(stocks.keys(), stocks.values()))))
print(max((zip(stocks.keys(), stocks.values()))))
print(sorted(zip(stocks.keys(), stocks.values())))

# Downloaded Pillow
from PIL import Image

img = Image.open("Azzaro.jpg")
print(img.size)

area = (25, 25, 25, 25)
cropped_img = img.crop(area)
# img.show()
cropped_img.show()
