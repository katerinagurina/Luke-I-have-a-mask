#by Andrei Erofeev
import numpy as np
import torch
import torch.nn as nn
import os
import warnings
import tqdm
import gc
import time
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, extract_face
import mmcv
import torchvision.transforms.functional as TF

import cv2

device = ('cuda' if torch.cuda.is_available else 'cpu')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_face(img, box, resize = 224, inverse = False):
    img = TF.to_tensor(img)
    box = [0 if b < 0 else b for b in box]
    face = img[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    face = TF.to_tensor(TF.resize(TF.to_pil_image(face), size = (resize, resize)))
    return face


class FaceDetector:
    def __init__(self, classificator, detector, detector_type='mtcnn'):
        self.classificator = classificator
        self.detector = detector
        self.detector_type = detector_type
        # self.classificator.to(device)

    def process_video(self, video_path, video_name):
        path = os.path.join(video_path, video_name)

        video = mmcv.VideoReader(path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        # self.classificator.load_state_dict(torch.load(self.classif_weights_path))

        # start tracking
        print('Start video face tracking')
        frames_tracked = []
        faces = {}
        boxes = {}
        for i, frame in enumerate(frames):
            print('\rTracking frame: {}'.format(i + 1), end='')
            # Detect faces
            if self.detector_type == 'mtcnn':
                frame_boxes, _ = self.detector.detect(frame)
            else:
                # torch_frame = TF.to_tensor(frame)
                new_image = TF.resize(frame, (320, 320))
                x_scale = 320 / frame.width
                y_scale = 320 / frame.height
                torch_frame = TF.to_tensor(new_image)
                torch_frame = TF.normalize(torch_frame, mean=mean, std=std)
                frame_boxes = self.detector(torch_frame.unsqueeze(0).float().to(device))[2]
            if frame_boxes is None:
                faces.update({str(i): None})
                boxes.update({str(i): None})
                frames_tracked.append(frame.resize((640, 360), Image.BILINEAR))
                continue
            # Draw faces
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            fnt = ImageFont.truetype("arial.ttf", size=24)
            frame_faces = []
            for box in frame_boxes:
                if self.detector_type != 'mtcnn':
                    box[0] /= x_scale
                    box[2] /= x_scale
                    box[1] /= y_scale
                    box[3] /= y_scale
                    # face = get_face(frame, box)
                # else:
                # face = extract_face(frame, box, image_size=224)#.transpose(1, 2)
                # face = extract_face(frame, box, image_size=224).transpose(1,2)
                face = get_face(frame, box)
                frame_faces.append(face)
                norm_face = TF.normalize(face, mean=mean, std=std)
                preds = torch.softmax(self.classificator(norm_face.unsqueeze(0).to(device)), dim=-1)
                if preds.argmax(-1)[0] == 0:
                    draw.rectangle(box.tolist(), outline=(0, 255, 0), width=4)
                    draw.text((box[0] - 15, box[1] - 20), "Mask", fill=(0, 255, 0, 255), font=fnt)
                elif preds.argmax(-1)[0] == 1:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
                    draw.text((box[0] - 15, box[1] - 20), "No mask", fill=(255, 0, 0, 255), font=fnt)
                else:
                    draw.rectangle(box.tolist(), outline=(0, 0, 255), width=4)
                    draw.text(((box[0] + box[2]) / 2 - 10, box[1] - 15), "Wrong mask", fill=(0, 0, 255, 255), font=fnt)

            # Add to frame list
            frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
            faces.update({str(i): frame_faces})
            boxes.update({str(i): boxes})
        print('\nDone')
        return frames_tracked, faces

    def create_video(self, frames, path='./', name='out.mp4', codec='MP4A', framerate=30, shape=(640, 360)):
        out_path = os.path.join(path, name)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), float(framerate), shape, True)
        for i in range(len(frames)):
            frame = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2BGR)
            out.write(frame)
        cv2.destroyAllWindows()
        out.release()
        return out

    def process_image(self, img, put_text=True):
        if self.detector_type == 'mtcnn':
            boxes, _ = self.detector.detect(img)
        else:
            # torch_frame = TF.to_tensor(frame)
            new_image = TF.resize(img, (320, 320))
            x_scale = 320 / img.width
            y_scale = 320 / img.height
            torch_frame = TF.to_tensor(new_image)
            torch_frame = TF.normalize(torch_frame, mean=mean, std=std)
            boxes = self.detector(torch_frame.unsqueeze(0).float().to(device))[2]
        if boxes is None:
            warnings.warn('No faces were detected!')
            return
            # Draw faces
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        fnt = ImageFont.truetype("arial.ttf", size=24)
        img_faces = []
        for box in boxes:
            if self.detector_type != 'mtcnn':
                box[0] /= x_scale
                box[2] /= x_scale
                box[1] /= y_scale
                box[3] /= y_scale
            face = get_face(img, box)
            img_faces.append(face)
            norm_face = TF.normalize(face, mean=mean, std=std)
            preds = torch.softmax(self.classificator(norm_face.unsqueeze(0).to(device)), dim=-1)
            if preds.argmax(-1)[0] == 0:
                if put_text:
                    draw.rectangle(box.tolist(), outline=(0, 255, 0), width=4)
                    draw.text((box[0] - 15, box[1] - 20), "Mask", fill=(0, 255, 0, 255), font=fnt)
                else:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
            elif preds.argmax(-1)[0] == 1:
                if put_text:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
                    draw.text((box[0] - 15, box[1] - 20), "No mask", fill=(255, 0, 0, 255), font=fnt)
                else:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
            else:
                if put_text:
                    draw.rectangle(box.tolist(), outline=(0, 0, 255), width=4)
                    draw.text(((box[0] + box[2]) / 2 - 10, box[1] - 15), "Wrong mask", fill=(0, 0, 255, 255), font=fnt)
                else:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=4)
        return img_draw, img_faces, boxes
