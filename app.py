from flask import Flask, request
import base64
import random
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.get_json())
    img = base64.b64decode(data['image_base64'].encode('utf8'))
    data['prediction'] = get_prediction(img)

    return json.dumps(data)

def get_prediction(img):
    """
    TODO THIS IS WHERE OUR ENTIRE PREDICTION CODE WILL GO
    """

    preds = [mock_pred() for i in range(3)]
    return preds

def mock_pred():
    return {
        'predicted_class': random.randint(1,4),
        'confidence': random.random(),
        'bounding_box': [random.random(), random.random(), random.random(), random.random()]
    }

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5000)
