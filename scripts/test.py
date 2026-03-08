import argparse
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from bluesky_scraper import print_progress

DATA_DIR = Path("../data")
IN_FILE = "hydrated_corpus.json"
OUT_FILE = "hydrated_corpus_captioned.json"

IN_PATH = DATA_DIR / IN_FILE
OUT_PATH = DATA_DIR / OUT_FILE

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"
OCR_TASK = "<OCR>"
CAPTION_TASK = "<DETAILED_CAPTION>"


class FlorenceRunner:
    def __init__(self, model_id=DEFAULT_MODEL_ID, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def run_batch(
        self,
        prompt: str,
        images: List[Image.Image],
        max_new_tokens: int = 128,
        beams: int = 1,
    ) -> List[Any]:
        prompts = [prompt] * len(images)
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                early_stopping=False,
            )

        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

        outputs = []
        for img, text in zip(images, texts):
            parsed = self.processor.post_process_generation(
                text,
                task=prompt,
                image_size=(img.width, img.height),
            )
            outputs.append(parsed.get(prompt, parsed))
        return outputs


def downscale(img: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    scale = max_side / s
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def load_image_from_url(url: str, timeout: int = 20, max_side: int = 768) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return downscale(img, max_side=max_side)


def extract_text_len(ocr_output: Any) -> int:
    if isinstance(ocr_output, str):
        return len(ocr_output.strip())

    if isinstance(ocr_output, dict):
        # Try common OCR output shapes
        if "text" in ocr_output and isinstance(ocr_output["text"], str):
            return len(ocr_output["text"].strip())
        if "labels" in ocr_output and isinstance(ocr_output["labels"], list):
            joined = " ".join(str(x) for x in ocr_output["labels"])
            return len(joined.strip())

    return 0


def normalize_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, str):
        return {"text": output}
    if isinstance(output, dict):
        return output
    return {"raw": output}


def build_jobs(corpus: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs = []
    data = corpus.get("data", [])

    for post_idx, post in enumerate(data):
        media = post.get("media") or {}
        images = media.get("images") or []
        videos = media.get("videos") or []
        post.setdefault("florence2", [])

        for media_idx, image_info in enumerate(images):
            url = image_info.get("fullsize") or image_info.get("thumb")
            if url:
                jobs.append({
                    "post_idx": post_idx,
                    "kind": "image",
                    "media_idx": media_idx,
                    "url": url,
                })

        for media_idx, video_info in enumerate(videos):
            url = video_info.get("thumbnail")
            if url:
                jobs.append({
                    "post_idx": post_idx,
                    "kind": "video_thumbnail",
                    "media_idx": media_idx,
                    "url": url,
                })

    return jobs


def producer_thread(
    jobs: List[Dict[str, Any]],
    ready_q: "queue.Queue[Dict[str, Any]]",
    max_side: int,
    num_download_workers: int,
):
    def worker(job: Dict[str, Any]) -> Dict[str, Any]:
        try:
            img = load_image_from_url(job["url"], max_side=max_side)
            return {**job, "image": img, "error": None}
        except (requests.RequestException, UnidentifiedImageError, OSError) as e:
            return {**job, "image": None, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=num_download_workers) as ex:
        for result in ex.map(worker, jobs):
            ready_q.put(result)

    ready_q.put({"_done": True})


def write_result(corpus: Dict[str, Any], item: Dict[str, Any], task: str, output: Any):
    post = corpus["data"][item["post_idx"]]
    post["florence2"].append({
        "kind": item["kind"],
        "media_idx": item["media_idx"],
        "source_url": item["url"],
        "task": task,
        "result": normalize_output(output),
    })


def write_error(corpus: Dict[str, Any], item: Dict[str, Any]):
    post = corpus["data"][item["post_idx"]]
    post["florence2"].append({
        "kind": item["kind"],
        "media_idx": item["media_idx"],
        "source_url": item["url"],
        "error": item["error"],
    })


def process_pipeline(
    corpus: Dict[str, Any],
    model_id: str,
    batch_size: int = 16,
    max_new_tokens: int = 128,
    beams: int = 1,
    max_side: int = 768,
    num_download_workers: int = 16,
    prefetch_qsize: int = 128,
    ocr_min_text: int = 12,
):
    jobs = build_jobs(corpus)
    runner = FlorenceRunner(model_id=model_id)

    ready_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=prefetch_qsize)
    prod = threading.Thread(
        target=producer_thread,
        args=(jobs, ready_q, max_side, num_download_workers),
        daemon=True,
    )
    prod.start()

    ocr_batch: List[Dict[str, Any]] = []
    caption_batch: List[Dict[str, Any]] = []
    result_cache: Dict[Tuple[str, str], Any] = {}
    done = False

    def flush_ocr():
        nonlocal ocr_batch, caption_batch
        if not ocr_batch:
            return

        uncached_items = []
        uncached_images = []

        for item in ocr_batch:
            key = (OCR_TASK, item["url"])
            if key in result_cache:
                output = result_cache[key]
                if extract_text_len(output) >= ocr_min_text:
                    write_result(corpus, item, OCR_TASK, output)
                else:
                    caption_batch.append(item)
            else:
                uncached_items.append(item)
                uncached_images.append(item["image"])

        if uncached_items:
            outputs = runner.run_batch(
                OCR_TASK,
                uncached_images,
                max_new_tokens=max_new_tokens,
                beams=beams,
            )
            for item, output in zip(uncached_items, outputs):
                result_cache[(OCR_TASK, item["url"])] = output
                if extract_text_len(output) >= ocr_min_text:
                    write_result(corpus, item, OCR_TASK, output)
                else:
                    caption_batch.append(item)

        ocr_batch = []

    def flush_caption():
        nonlocal caption_batch
        if not caption_batch:
            return

        uncached_items = []
        uncached_images = []

        for item in caption_batch:
            key = (CAPTION_TASK, item["url"])
            if key in result_cache:
                write_result(corpus, item, CAPTION_TASK, result_cache[key])
            else:
                uncached_items.append(item)
                uncached_images.append(item["image"])

        if uncached_items:
            outputs = runner.run_batch(
                CAPTION_TASK,
                uncached_images,
                max_new_tokens=max_new_tokens,
                beams=beams,
            )
            for item, output in zip(uncached_items, outputs):
                result_cache[(CAPTION_TASK, item["url"])] = output
                write_result(corpus, item, CAPTION_TASK, output)

        caption_batch = []

    while not done:
        item = ready_q.get()

        if item.get("_done"):
            done = True
            break

        if item["error"] is not None:
            write_error(corpus, item)
            continue

        ocr_batch.append(item)

        if len(ocr_batch) >= batch_size:
            flush_ocr()

        if len(caption_batch) >= batch_size:
            flush_caption()

    flush_ocr()
    flush_caption()
    prod.join()

    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=IN_PATH)
    parser.add_argument("--out", dest="out_path", default=OUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=128)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--ocr-min-text", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--beams", type=int, default=1)
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    updated = process_pipeline(
        corpus=corpus,
        model_id=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        beams=args.beams,
        max_side=args.max_side,
        num_download_workers=args.download_workers,
        prefetch_qsize=args.prefetch,
        ocr_min_text=args.ocr_min_text,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()