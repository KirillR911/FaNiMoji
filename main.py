
from fastapi import FastAPI, File, UploadFile, Form,HTTPException
from pydantic import BaseModel
from typing import List
import io
import numpy as np
import sys
import logging
from model import Prediction
from facefeature import Face
import json
import os
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler('test.log')
fileHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[QUERY] %(asctime)s -  %(image_name)s - %(srv)s - %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)


app = FastAPI()

@app.get('/')
def root_route():
  return { 'error': 'Use POST /prediction instead of the root route!' }

@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...), srv : str = Form(...)):
  if file.content_type.startswith('image/') is False:
    extr = {'srv': srv, 'image_name': file.filename}
    logger.error("ERROR", extra=extr)
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')
  try:
    contents = await file.read()
    print("iMAGE")
    pil_image = Image.open(io.BytesIO(contents))
    # print()
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')
    pim = Face(pil_image,label=file.filename)
    data = pim.get_params()
    extr = {'srv' : srv, 'image_name' : file.filename}
    # print(extr)
    logger.info("OK", extra=extr)
    
    return {
      'filename'   : file.filename,
      'contenttype': file.content_type,
      'prediction' : data
    }
  except:
    e = sys.exc_info()[1]
    extr = {'srv': srv, 'image_name': None}
    logger.error(f"ERROR: {str(e)}", extra=extr)
    raise HTTPException(status_code=500, detail=str(e))