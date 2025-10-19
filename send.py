import requests

url = "https://greenscan-api-2.onrender.com/predict"
# Send 1-4 images with the key 'images'
files = [
    ('images', open('test.jpeg', 'rb'))
    # Add more images if you want to test multiple:
    # ('images', open('test2.jpeg', 'rb')),
    # ('images', open('test3.jpeg', 'rb')),
]
response = requests.post(url, files=files)
print(response.json())