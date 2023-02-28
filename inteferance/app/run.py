from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
from training.src.make_data_train import Trainer
from training.src.face_rec import Face_recognition
import numpy as np
from PIL import Image
import cv2
import base64
import io


face_recogny = Face_recognition()
app = FastAPI()

# Class 
class Member(BaseModel):
    name: str
    image: str

class Detection(BaseModel):
    image: str

class Elimination(BaseModel):
    name: str

# Function encode to base64 and decode from base64
def stringToRGB(base64_string):
    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def RGB2String(image):
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

# Router
@app.post("/add")
async def add_new_member(item: Member):
    trainer = Trainer()
    img = stringToRGB(item.image)
    trainer.add_member(image = img, name = item.code)
    return "Add member successfully!"

@app.delete("/delete")
async def delete_member(item: Elimination):
    return item.name

@app.post("/detection")
async def detection_face(item: Detection):
    #image = base64.b64encode(item.image)
    image = stringToRGB(item.image)
    result = face_recogny.recogny_face(image)
    frame, bbox, name, current_time = result
    frame_b64 = RGB2String(frame)
    return  name, current_time


if __name__ == "__main__":
    uvicorn.run(app)