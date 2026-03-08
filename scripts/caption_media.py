import argparse
import json
import re
import time
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
from PIL import Image

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

DATA_DIR = Path("../data")
IN_FILE = "hydrated_corpus.json"
OUT_FILE = "hydrated_corpus_captioned.json"

IN_PATH = DATA_DIR / IN_FILE
OUT_PATH = DATA_DIR / OUT_FILE

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
DEFAULT_CAPTION_TASK = "<DETAILED_CAPTION>"
DEFAULT_OCR_TASK = "<OCR>"

OCR_TRIGGER_KEYWORDS = {
    "screenshot", "tweet", "post", "headline", "article", "text", "meme", "quote", "message",
    "email", "document", "statement", "caption", "chart", "graph", "infographic", "news",
    "facebook", "instagram", "reddit", "dm", "thread", "tumblr", "twitter", "4chan", "linkedin"
}

SHORT_TEXT_LEN = 20

class FlorenceCaption:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_ID, 
                                                          torch_dtype=torch.float16, 
                                                          trust_remote_code=True, 
                                                          low_cpu_mem_usage=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID, trust_remote_code=True)

    def caption_image(self, prompt, image: Image.Image, max_new_tokens=96, beams=1) -> str:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(dtype=self.model.dtype),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                early_stopping=False)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        return parsed[prompt]
    
    def caption_batch(self, prompt, images,  max_new_tokens: int = 96, beams: int = 1) -> List[str]:
        prompts = [prompt] * len(images)
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(dtype=self.model.dtype),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                early_stopping=False)
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

        out: List[str] = []
        for img, gen_text in zip(images, generated_texts):
            parsed = self.processor.post_process_generation(
                gen_text,
                task=prompt,
                image_size=(img.width, img.height),
            )
            out.append(parsed[prompt])
        return out

def pil_from_url(url: str, timeout=20) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def downscale(img, max_side=768):
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    scale = max_side / s
    return img.resize((int(w*scale), int(h*scale)))

if __name__ == "__main__":
    cap = FlorenceCaption()
    print("loaded:", DEFAULT_MODEL_ID)
    print("transformers device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    with IN_PATH.open("r") as f:
        corpus = json.load(f)
        data = corpus.get("data")
    out = {}
    img_batch = []
    vid_batch = []
    for i in data:
        media = i.get("media")
        images = media.get("images")
        videos = media.get("videos")
        img_flag = False
        vid_flag = False
        if len(img_batch) < 25:
            img_flag = True
            for i in images:
                img_batch.append( downscale(pil_from_url(i.get("fullsize"))) )
        if len(vid_batch) < 25:
            vid_flag = True
            for i in videos:
                vid_batch.append( downscale(pil_from_url(i.get("thumbnail"))) )
        if not img_flag:
            img_caps = cap.caption_batch(DEFAULT_CAPTION_TASK, img_batch)
            img_batch = []
        if not vid_flag:
            vid_caps = cap.caption_batch(DEFAULT_CAPTION_TASK, vid_batch)
            vid_batch = []
    if img_batch:
        img_caps = cap.caption_batch(DEFAULT_CAPTION_TASK, img_batch)
        img_batch = []
    if vid_batch:
        vid_caps = cap.caption_batch(DEFAULT_CAPTION_TASK, vid_batch)
        vid_batch = []