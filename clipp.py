import os
import cv2
from pathlib import Path
from PIL import Image
import torch
import clip
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title='CLIP Visual Analysis Service')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
torch.set_num_threads(1)
model, preprocess = clip.load('RN50', device=device)

text_prompts = ['teacher gesturing', 'pupils engaging', 'pupils attentive']
text_inputs = clip.tokenize(text_prompts).to(device)


def extract_frames(video_path, output_folder, interval=150):
    output_folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count, saved_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = output_folder / f'frame_{saved_count:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    return sorted(output_folder.glob('*.jpg'))


@app.post('/analyse')
async def analyse_clip(video: UploadFile = File(...)):
    temp_video = Path('temp_video.mp4')
    with open(temp_video, 'wb') as f:
        f.write(await video.read())

    frame_dir = Path('frames_tmp')
    frame_paths = extract_frames(temp_video, frame_dir, interval=150)

    analysis = {label: [] for label in text_prompts}
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path)
                image_input = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                probs = (image_features @ text_features.T).softmax(dim=-1)[0]
                for i, label in enumerate(text_prompts):
                    analysis[label].append(probs[i].item())
            except Exception:
                continue

    avg_scores = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in analysis.items()}

    for f in frame_dir.glob('*.jpg'):
        f.unlink()
    frame_dir.rmdir()
    temp_video.unlink()


    return JSONResponse(content={'visual_analysis': avg_scores})
