from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame, get_face_swapper, swap_face
from modules.face_analyser import get_one_face, get_many_faces
from modules.utilities import resolve_relative_path
import insightface
import modules.globals
import onnxruntime as ort

bp = Blueprint('image', __name__)
logger = logging.getLogger(__name__)

def enhance_face(image, face):
    # Get face region
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_region = image[y1:y2, x1:x2]
    
    # Apply bilateral filter for noise reduction while preserving edges
    face_region = cv2.bilateralFilter(face_region, 9, 75, 75)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    face_region = cv2.filter2D(face_region, -1, kernel)
    
    # Put enhanced face back
    image[y1:y2, x1:x2] = face_region
    return image

@bp.route('/image_swap')
def image_swap():
    return render_template('image_swap.html')

@bp.route('/swap_faces', methods=['POST'])
def swap_faces():
    try:
        if 'face' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Missing face or target image'}), 400

        face_file = request.files['face']
        target_file = request.files['target']

        if face_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read images
        face_image = cv2.imdecode(np.frombuffer(face_file.read(), np.uint8), cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if face_image is None or target_image is None:
            return jsonify({'error': 'Failed to read image files'}), 400

        # Store original size
        original_size = target_image.shape[:2]
        
        # Resize to optimal size for processing
        target_image = cv2.resize(target_image, Config.MIN_FRAME_SIZE, interpolation=cv2.INTER_LINEAR)

        # Force CUDA provider
        modules.globals.execution_providers = ['CUDAExecutionProvider']
        
        # Get source face using Deep Live Cam's face detection
        logger.info("Detecting source face...")
        source_face = get_one_face(face_image)
        if not source_face:
            return jsonify({'error': 'No face detected in source image'}), 400
        logger.info(f"Source face detected with bbox: {source_face.bbox}")

        # Get target face
        logger.info("Detecting target face...")
        target_face = get_one_face(target_image)
        if not target_face:
            return jsonify({'error': 'No face detected in target image'}), 400
        logger.info(f"Target face detected with bbox: {target_face.bbox}")

        # Process the frame using Deep Live Cam's swap_face
        logger.info("Starting face swap processing...")
        try:
            # Use the exact same swap_face function from Deep Live Cam
            processed_image = swap_face(source_face, target_face, target_image)
            
            if processed_image is None:
                return jsonify({'error': 'Face swap processing failed'}), 500
                
            logger.info("Face swap completed successfully")
            
            # Resize back to original size with high quality
            processed_image = cv2.resize(processed_image, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            logger.error(f"Error in face swap processing: {str(e)}")
            return jsonify({'error': f'Face swap processing error: {str(e)}'}), 500

        # Convert to base64 with maximum quality
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_base64})

    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return jsonify({'error': str(e)}), 500 