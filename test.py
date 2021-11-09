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
        "image_base64": image_base64,
        "mask": True
    }

    response = requests.post(url, json=json.dumps(data), headers=header)
    # response = requests.post(url, "random")
    return response.json()

if __name__ == '__main__':    
    url = 'http://127.0.0.1:5000/predict' # TODO replace with endpoint after deployment
    path = 'image/1.jpg'
    output = predict(url, path)
    print(json.dumps(output, indent=4, sort_keys=False))



# curl -X POST -H "Content-Type: application/json" -d "{\"filename\":\"1.jpg\"}" http://localhost:5000/predict