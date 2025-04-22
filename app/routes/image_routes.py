from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
import time
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face
import modules.globals

bp = Blueprint('image', __name__)
logger = logging.getLogger(__name__)

# Global variables for frame processing
last_processed_frame = None
processing_enabled = True
FACE_CACHE_DURATION = 0.1

@bp.route('/image_swap')
def image_swap():
    return render_template('image_swap.html')

@bp.route('/swap_faces', methods=['POST'])
def swap_faces():
    global last_processed_frame, processing_enabled
    
    try:
        if 'face' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Missing face or target image'}), 400

        face_file = request.files['face']
        target_file = request.files['target']

        if face_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read images with maximum quality
        face_image = cv2.imdecode(np.frombuffer(face_file.read(), np.uint8), cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if face_image is None or target_image is None:
            return jsonify({'error': 'Failed to read image files'}), 400

        # Set execution providers
        modules.globals.execution_providers = ['CUDAExecutionProvider']
        
        # Get source face
        logger.info("Detecting source face...")
        source_face = get_one_face(face_image)
        if not source_face:
            return jsonify({'error': 'No face detected in source image'}), 400
        logger.info(f"Source face detected with bbox: {source_face.bbox}")

        # Process the frame using Deep Live Cam's process_frame
        logger.info("Starting face swap processing...")
        try:
            # Process the frame
            processed_image = process_frame(source_face, target_image)
            
            if processed_image is None:
                if last_processed_frame is not None:
                    processed_image = last_processed_frame.copy()
                else:
                    return jsonify({'error': 'Face swap processing failed'}), 500
            
            # Cache the processed frame
            last_processed_frame = processed_image.copy()
            processing_enabled = True
            
            logger.info("Face swap completed successfully")
            
        except Exception as e:
            logger.error(f"Error in face swap processing: {str(e)}")
            if last_processed_frame is not None:
                processed_image = last_processed_frame.copy()
            else:
                return jsonify({'error': f'Face swap processing error: {str(e)}'}), 500
            processing_enabled = False

        # Convert to base64 with maximum quality
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_base64})

    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return jsonify({'error': str(e)}), 500 