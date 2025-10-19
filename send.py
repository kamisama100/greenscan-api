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
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
if response.status_code == 200:
    print(f"JSON: {response.json()}")
else:
    print("Error: Non-200 status code")