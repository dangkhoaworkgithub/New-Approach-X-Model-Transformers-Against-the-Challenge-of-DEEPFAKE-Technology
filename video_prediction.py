import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
import argparse
from load_model import load_model_xmt
from PIL import ImageFont, ImageDraw
from deepface import DeepFace

mtcnn, model, device = load_model_xmt()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def draw_box_and_label(image, box, label):
    draw = ImageDraw.Draw(image)
    box = [int(coordinate) for coordinate in box]
    expanded_box = [box[0], box[1] - 20, box[2], box[3] + 20]
    box_tuple = (expanded_box[0], expanded_box[1], expanded_box[2], expanded_box[3])

    color = "red" if label.startswith("Fake") else "yellow"

    font = ImageFont.load_default(30)
    draw.rectangle(box_tuple, outline=color, width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill=color, font=font)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def process_frame(frame):
    image = Image.fromarray(frame)
    detections = DeepFace.detectFace(image, detector_backend = 'mtcnn')

    for detection in detections:
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        box = (x, y, x+w, y+h)
        face = image.crop(box)
        face = np.array(face)
        face = normalize_transform(face).unsqueeze(0).to(device)

        prediction = model(face)
        prediction = torch.softmax(prediction, dim=1)
        pred_real_percentage = prediction[0][1].item() * 100
        pred_fake_percentage = prediction[0][0].item() * 100

        if max(pred_real_percentage, pred_fake_percentage) > 80:
            _, predicted_class = torch.max(prediction, 1)
            pred_label = predicted_class.item()
            label = "Real" if pred_label == 1 else "Fake"
        else:
            label = "Calculating"

        if label == "Calculating":
            label_with_probabilities = f"{label}"
        elif label == "Fake":
            label_with_probabilities = f"{label}:{(100 - pred_real_percentage):.2f}%"
        else:
            label_with_probabilities = f"{label}:{pred_real_percentage:.2f}%"
        draw_box_and_label(image, box, label_with_probabilities)

    return np.array(image)

def save_video(frames, output_path, fps=20.0, resolution=(1280, 720)):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def resize_frame(frame, target_size=(1920, 1080)):
    """
    Resize the frame to the target size while maintaining aspect ratio.
    Adds padding if necessary to fit the target size without distortion.
    """
    h, w = frame.shape[:2]
    desired_w, desired_h = target_size

    ratio_w = desired_w / w
    ratio_h = desired_h / h
    new_w, new_h = w, h

    if ratio_w < ratio_h:
        new_w = desired_w
        new_h = round(h * ratio_w)
        frame = cv2.resize(frame, (new_w, new_h))
        pad_top = (desired_h - new_h) // 2
        pad_bottom = desired_h - new_h - pad_top
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT)
    else:
        new_h = desired_h
        new_w = round(w * ratio_h)
        frame = cv2.resize(frame, (new_w, new_h))
        pad_left = (desired_w - new_w) // 2
        pad_right = desired_w - new_w - pad_left
        frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return frame

