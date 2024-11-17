from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
import os
import cv2
import threading
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# YOLO 모델 경로 설정
model_path = '/Users/eden/Downloads/cross5/1108_finetuned.pt'  # 실제 모델 파일 경로로 수정
model = YOLO(model_path)

# 미리 준비된 비디오 목록
PRESET_VIDEOS = {
    "video1": "static/videos/video1.mp4",
    "video2": "static/videos/video2.mp4",
    "video3": "static/videos/video3.mp4",
    "video4": "static/videos/video4.mp4",
}

# 진행 상태 변수 (0~100%)
processing_progress = 0

@app.route("/")
def index():
    # 원본 비디오와 처리된 비디오의 경로 설정
    original_video_url = request.args.get('original_video_url')
    processed_video_url = request.args.get('processed_video_url')

    return render_template("index.html",
                           original_video_url=original_video_url,
                           processed_video_url=processed_video_url,
                           videos=PRESET_VIDEOS)

@app.route("/process", methods=["POST"])
def process_video():
    global processing_progress
    # 진행 상태 초기화
    processing_progress = 0

    # 선택된 동영상 키 받기
    video_key = request.form.get("video_key")

    if video_key not in PRESET_VIDEOS:
        return "유효한 비디오를 선택해야 합니다.", 400

    # 선택된 동영상의 경로 가져오기
    original_video_path = PRESET_VIDEOS[video_key]
    original_video_url = url_for('static', filename=os.path.basename(original_video_path))

    # 검출 결과 저장할 임시 비디오 파일 경로
    output_path = os.path.join('static', os.path.basename(original_video_path).replace(".mp4", "_output.mp4"))

    # 비디오 처리 시작
    cap = cv2.VideoCapture(original_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 프레임별로 객체 검출 수행
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        # 검출된 객체에 대해 바운딩 박스 그리기
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence < 0.3:  # 기본 신뢰도 기준 0.3
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

        # 처리 진행 상태 업데이트
        frame_count += 1
        processing_progress = int((frame_count / total_frames) * 100)

    cap.release()
    out.release()

    # 비디오 재인코딩
    reencoded_output_path = os.path.join('static', os.path.basename(output_path).replace(".mp4", "_reencoded.mp4"))
    reencode_video(output_path, reencoded_output_path)

    # 처리된 비디오 URL 반환
    processed_video_url = url_for('static', filename=os.path.basename(reencoded_output_path))

    return redirect(f"/?original_video_url={original_video_url}&processed_video_url={processed_video_url}")

def reencode_video(input_path, output_path):
    """moviepy를 사용하여 비디오를 H.264와 AAC 코덱으로 재인코딩"""
    # VideoFileClip을 사용하여 비디오 로드
    clip = VideoFileClip(input_path)
    
    # 비디오를 H.264 코덱, 오디오를 AAC 코덱으로 저장
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

@app.route("/get_progress")
def get_progress():
    return jsonify({"progress": processing_progress})

if __name__ == "__main__":
    app.run(debug=True)