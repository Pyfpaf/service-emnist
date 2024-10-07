import numpy as np
import torch
import pickle

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles

from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()
with open('myapp/model.pkl', 'rb') as f:
    data = pickle.load(f)
model.load_state_dict(data)

# make symbol dict
with open('emnist-balanced-mapping.txt', 'r') as file:
    content = file.read()
lines = content.strip().split('\n')
labels_dict = {int(lines[i].split(' ')[0]): chr(int(lines[i].split(' ')[1])) for i in range(len(lines))}

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    image = np.array(list(map(int, image[1:-1].split(','))))
    x = torch.from_numpy(image.reshape(-1, 28 * 28).astype(np.float32))
    model.eval()
    with torch.no_grad():
        pred = model(x)
    return {'prediction': labels_dict[pred.argmax().item()]}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
