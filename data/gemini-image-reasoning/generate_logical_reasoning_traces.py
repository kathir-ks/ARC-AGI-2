#import libraries
# %pip install -U -q "google-genai>=1.0.0"

from PIL import Image as PILImage
import time
from dotenv import load_dotenv

import os
from google import genai
from google.genai import types
import argparse
import json

# load environment variables

load_dotenv()

# inspired from https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Tag_and_caption_images.ipynb#scrollTo=NwP4PBGWoQiJ

def list_images(folder_path):
    # list all the images 
    all_entires = os.listdir(folder_path)
    files = [entry for entry in all_entires if os.path.isfile(os.path.join(folder_path, entry))]
    files = [os.path.join(folder_path, file_path) for file_path in files]
    return files

MODEL_ID='gemini-2.5-pro' # @param ["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true, isTemplate: true}

# a helper function for calling

def generate_text_using_image(client, prompt, image_path, sleep_time=4):
  start = time.perf_counter()
  response = client.models.generate_content(
    model=MODEL_ID,
    contents=[PILImage.open(image_path)],
    config=types.GenerateContentConfig(
        system_instruction=prompt
    ),
)
  end = time.perf_counter()
  duration = end - start
  time.sleep(sleep_time - duration if duration < sleep_time else 0)
  return response.text

def setup_genai_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

reasoning_prompt ="""
     You are an agent that tries to understand the ARC-AGI Competition dataset. 
     The input size is 30 x 30 grids, and the output is 30 x 30 dimension. 
     Both the input and the output images contains different objects of different shapes. 
     The most important way to identify objects/patterns in using the color and the 8 
     directional connection between the grids. The output might have some reduced dimensions 
     as well. The output objects will have some relation to the input objects in some way or the other.
     The relation between the input and the output will be co-related by the colors, 
     similar structure or shape of the objects. 

     Action Item:
     1. Generate Logical reasoning for the transformation of the input to the output. 
        a. Try to think much longer before arriving at a conclusion. 
        b. Do reasoning in more detailed and structured steps.
            i). Identify the objects, shapes, patterns in both the input and the output. 
            ii). Try to find the transformation and the relation between them. 
            iii). Do the reasoning in multiple steps. and no need to find the reason in a single steps. 
            For example, you can do the thinking implicity, try to map patterns, by comparing different 
            objects together. 
        c. Finally generate a pseudo code for the transformation.

    Note to keep in mind: Don't rush to find the answer, try to do it in multiple steps in a more
    detaile way. Also try to self correct if you are wrong at certain places. 
"""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY"),required=False)
    parser.add_argument("--images_src_path", type=str, required=True)

    args = parser.parse_args()

    api_key = args.api_key
    images_src_path = args.images_src_path

    client = setup_genai_client(api_key)

    # load the images
    files = list_images(images_src_path)

    if len(files) <= 0 :
        raise ValueError("No of files should not be zero")
    
    entry = 0
    # Reasoning traces
    reasoning_traces = {}
    # generate the traces from the prompt
    for file in files:
        task_type = 'train'
        image_name = file.split('\\')[-1]
        task_id = image_name.split('_')[0]
        if 'test' in file:
            task_type = 'test'
        logical_reasoning = generate_text_using_image(client, reasoning_prompt, file)

        task_set = {
            'task_type': task_type, 
            'image_name': image_name, 
            'task_id': task_id, 
            'reasoning': logical_reasoning
        }
    
        reasoning_traces[entry] = task_set
        entry += 1
        break
    
    with open('reasoning_traces.json', 'w') as f:
        json.dump(reasoning_traces, f)