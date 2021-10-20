import os
import json
import base64
import requests

def predict(url, image_filepath):
    image = open(image_filepath, "rb")
    image = image.read()
    image_base64 = base64.b64encode(image).decode('utf8')

    image_filename = os.path.basename(image_filepath)

    header = {"content-type": "application/json"}
    data = {
        "filename": image_filename,
        "image_base64": image_base64[:100] # TODO truncated for now cos v long
    }

    response = requests.post(url, json=json.dumps(data), headers=header)
    return response.json()

if __name__ == '__main__':    
    url = 'http://localhost:5000/predict' # TODO replace with endpoint after deployment
    path = 'image/1.jpg'
    print(json.dumps(predict(url, path), indent=4, sort_keys=False))