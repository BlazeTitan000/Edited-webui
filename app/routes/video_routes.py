from flask import Blueprint, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame

bp = Blueprint('video', __name__)
logger = logging.getLogger(__name__)

@bp.route('/video_swap')
def video_swap():
    return render_template('video_swap.html')

@bp.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'face' not in request.files or 'video' not in request.files:
            return jsonify({'error': 'Missing face image or video'}), 400

        face_file = request.files['face']
        video_file = request.files['video']
        provider = request.form.get('provider', 'CUDAExecutionProvider')

        if face_file.filename == '' or video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read face image
        face_image = cv2.imdecode(np.frombuffer(face_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Save video temporarily
        temp_video_path = 'temp_video.mp4'
        video_file.save(temp_video_path)

        # Process video
        output_path = 'output_video.mp4'
        process_video_frames(face_image, temp_video_path, output_path, provider)

        # Clean up temp file
        os.remove(temp_video_path)

        # Return processed video
        return send_file(output_path, mimetype='video/mp4')

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

def process_video_frames(face_image, input_path, output_path, provider):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = process_frame(face_image, frame, provider)
        out.write(processed_frame)

    cap.release()
    out.release() 