from flask import Blueprint, render_template, request, Response
import cv2
import numpy as np
import base64
import time
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame
import os
import io
from PIL import Image

bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Global variables
source_face = None
recording = False
out = None
last_processed_frame = None
processing_enabled = True

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/upload_face', methods=['POST'])
def upload_face():
    global source_face
    
    if 'face' not in request.files:
        return "No file part", 400
    
    file = request.files['face']
    if file.filename == '':
        return "No selected file", 400
    
    try:
        # Save uploaded file
        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Initialize CUDA session
        start_time = time.time()
        session = get_onnx_session()
        if session is None:
            return "Error initializing face processing with CUDA", 500
        logger.info(f"CUDA session initialization time: {time.time() - start_time:.3f} seconds")
        
        # Pre-process the source face
        start_time = time.time()
        source_face = get_one_face_optimized(cv2.imread(file_path))
        if source_face is None:
            return "No face detected in the image", 400
        logger.info(f"Face detection time: {time.time() - start_time:.3f} seconds")
            
        logger.info("Face image processed successfully with CUDA")
        return "Success", 200
        
    except Exception as e:
        logger.error(f"Error processing face image: {e}")
        return str(e), 500

@bp.route('/process_frame', methods=['POST'])
def handle_frame():
    global source_face, recording, out, last_processed_frame, processing_enabled
    
    if source_face is None:
        return "No face image uploaded. Please upload a face image first.", 400

    try:
        # Get frame data from request
        frame_data = request.get_data()
        if not frame_data:
            return "No frame data received", 400

        # Convert received data to OpenCV image
        frame = np.array(Image.open(io.BytesIO(frame_data)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Store original frame size
        original_size = frame.shape[:2]
        
        # Resize frame to minimum size for processing
        frame = cv2.resize(frame, Config.MIN_FRAME_SIZE)
        
        # Process the frame with timing
        start_time = time.time()
        
        # Use optimized face detection
        target_face = get_one_face_optimized(frame)
        if target_face:
            try:
                # Process the frame
                processed_frame = process_frame(source_face, frame)
                
                # Resize back to original size
                processed_frame = cv2.resize(processed_frame, (original_size[1], original_size[0]))
                
                # Add black rectangle only when processing is successful
                if Config.BLACK_RECTANGLE_ENABLED:
                    x, y = Config.BLACK_RECTANGLE_POSITION
                    w, h = Config.BLACK_RECTANGLE_SIZE
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
                
                # Cache the processed frame
                last_processed_frame = processed_frame.copy()
                processing_enabled = True
                
            except Exception as e:
                logger.error(f"Error in face processing: {e}")
                if last_processed_frame is not None:
                    processed_frame = last_processed_frame.copy()
                else:
                    processed_frame = frame
                processing_enabled = False
        else:
            if last_processed_frame is not None:
                processed_frame = last_processed_frame.copy()
            else:
                processed_frame = frame
            processing_enabled = False
        
        processing_time = time.time() - start_time
        logger.info(f"Frame processing time: {processing_time:.3f} seconds")

        if recording and out:
            out.write(processed_frame)

        # Convert processed frame to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')

        return processed_frame_data, 200

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        if last_processed_frame is not None:
            _, buffer = cv2.imencode('.jpg', last_processed_frame)
            processed_frame_data = base64.b64encode(buffer).decode('utf-8')
            return processed_frame_data, 200
        return str(e), 500

@bp.route('/record/start', methods=['POST'])
def start_record():
    global recording, out
    try:
        recording = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'output_{timestamp}.mp4'
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'X264'), 20.0, (640, 480))
        logger.info(f"Recording started: {filename}")
        return '', 204
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return str(e), 500

@bp.route('/record/stop', methods=['POST'])
def stop_record():
    global recording, out
    try:
        recording = False
        if out:
            out.release()
            out = None
        logger.info("Recording stopped")
        return '', 204
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return str(e), 500 