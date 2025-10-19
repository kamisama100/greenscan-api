import requests

url = "https://greenscan-api.onrender.com/predict"
files = {'image': open('test.jpeg', 'rb')}
response = requests.post(url, files=files)
print(response.json())