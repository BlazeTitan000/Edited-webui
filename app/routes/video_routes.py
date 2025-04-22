from flask import Blueprint, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import logging
import time
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face
import modules.globals

bp = Blueprint('video', __name__)
logger = logging.getLogger(__name__)

# Global variables for frame processing
last_processed_frame = None
processing_enabled = True
FACE_CACHE_DURATION = 0.1

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

        if face_file.filename == '' or video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create temporary directory for processing
        temp_dir = os.path.join(Config.UPLOAD_FOLDER, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files
        face_path = os.path.join(temp_dir, 'face.jpg')
        video_path = os.path.join(temp_dir, 'input.mp4')
        output_path = os.path.join(temp_dir, 'output.mp4')

        face_file.save(face_path)
        video_file.save(video_path)

        # Read face image
        face_image = cv2.imread(face_path)
        if face_image is None:
            return jsonify({'error': 'Failed to read face image'}), 400

        # Set execution providers
        modules.globals.execution_providers = ['CUDAExecutionProvider']
        
        # Get source face
        logger.info("Detecting source face...")
        source_face = get_one_face(face_image)
        if not source_face:
            return jsonify({'error': 'No face detected in source image'}), 400
        logger.info(f"Source face detected with bbox: {source_face.bbox}")

        # Process video
        logger.info("Starting video processing...")
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = process_frame(source_face, frame)
                
                if processed_frame is None:
                    if last_processed_frame is not None:
                        processed_frame = last_processed_frame.copy()
                    else:
                        processed_frame = frame
                
                # Cache the processed frame
                last_processed_frame = processed_frame.copy()
                processing_enabled = True
                
                # Write processed frame
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log progress every 30 frames
                    logger.info(f"Processed {frame_count} frames")

            cap.release()
            out.release()
            
            logger.info("Video processing completed successfully")

            # Return the processed video
            return send_file(output_path, as_attachment=True, download_name='processed_video.mp4')

        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return jsonify({'error': f'Video processing error: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return jsonify({'error': str(e)}), 500 