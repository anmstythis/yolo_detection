from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def video_check(video_path):
    model = YOLO('yolov8m-seg.pt')

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print('Невозможно прочесть видео')
        return
    
    while True:
        ret, frame = video.read()

        if not ret:
            print('Кадр не считан')
            break

        frame_resized = cv2.resize(frame, (500,500))

        detect_frame = model(frame_resized, imgsz=640, iou=0.4, conf=0.8, verbose=True)

        result = detect_frame[0]

        annotated_frame = result.plot()

        cv2.imshow('Image detected',  annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection video YOLO')
    parser.add_argument('video_path', type=str, help='Path to video')
