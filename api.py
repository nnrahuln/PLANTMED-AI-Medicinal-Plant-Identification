import os, io, pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

BASE=os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE,'class_indices.pkl'),'rb') as f: ci=pickle.load(f)
idx_to_class={int(v):k for k,v in ci.items()}
base=keras.applications.InceptionV3(weights=None,include_top=False,input_shape=(224,224,3))
model=keras.Sequential([base,layers.GlobalAveragePooling2D(),layers.Dense(256,activation='relu'),layers.Dropout(.5),layers.Dense(128,activation='relu'),layers.Dropout(.3),layers.Dense(len(idx_to_class),activation='softmax')])
model.build((None,224,224,3)); model.load_weights(os.path.join(BASE,'model_weights.weights.h5'))
app=FastAPI(title='PLANTMED API',version='1.0.0')

@app.get('/health')
def health(): return {'status':'ok','classes':len(idx_to_class)}

@app.post('/predict')
async def predict(file: UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith('image/'): raise HTTPException(400,'Upload an image file.')
    raw=await file.read()
    try: img=Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception as e: raise HTTPException(400,f'Invalid image: {e}')
    arr=keras_image.img_to_array(img.resize((224,224))); arr=preprocess_input(np.expand_dims(arr,0)); p=model.predict(arr,verbose=0)[0]
    ids=np.argsort(p)[::-1][:5]
    return {'prediction':idx_to_class[int(ids[0])],'confidence':round(float(p[ids[0]])*100,2),'top5':[{'name':idx_to_class[int(i)],'confidence':round(float(p[i])*100,2)} for i in ids]}
