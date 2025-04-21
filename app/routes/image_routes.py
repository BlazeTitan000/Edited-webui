from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame, get_face_swapper
from modules.face_analyser import get_one_face, get_face_analyser
from modules.utilities import resolve_relative_path
import insightface
import modules.globals
import onnxruntime as ort

bp = Blueprint('image', __name__)
logger = logging.getLogger(__name__)

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
        provider = request.form.get('providers', 'CUDAExecutionProvider')

        if face_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read images
        face_image = cv2.imdecode(np.frombuffer(face_file.read(), np.uint8), cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if face_image is None or target_image is None:
            return jsonify({'error': 'Failed to read image files'}), 400

        # Set execution providers for face analyzer
        modules.globals.execution_providers = [provider]
        
        # Initialize face analyzer with CUDA support
        logger.info("Initializing face analyzer with CUDA...")
        face_analyzer = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider']  # Force CUDA only
        )
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        if face_analyzer is None:
            return jsonify({'error': 'Failed to initialize face analyzer'}), 500

        # Get source face with detailed logging
        logger.info("Detecting face in source image...")
        source_faces = face_analyzer.get(face_image)
        if not source_faces:
            return jsonify({'error': 'No face detected in source image'}), 400
        source_face = source_faces[0]  # Use the first face found
        logger.info(f"Source face detected with bbox: {source_face.bbox}")

        # Get target face with detailed logging
        logger.info("Detecting face in target image...")
        target_faces = face_analyzer.get(target_image)
        if not target_faces:
            return jsonify({'error': 'No face detected in target image'}), 400
        target_face = target_faces[0]  # Use the first face found
        logger.info(f"Target face detected with bbox: {target_face.bbox}")

        # Initialize face swapper with CUDA support
        logger.info("Initializing face swapper with CUDA...")
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        face_swapper = insightface.model_zoo.get_model(
            model_path,
            providers=['CUDAExecutionProvider']  # Force CUDA only
        )
        
        if face_swapper is None:
            return jsonify({'error': 'Failed to initialize face swapper'}), 500

        # Process face swap with detailed logging
        logger.info("Starting face swap processing...")
        try:
            # Perform face swap with enhanced quality settings
            processed_image = face_swapper.get(
                target_image, 
                target_face, 
                source_face, 
                paste_back=True,
                upsample=True  # Enable upsampling for better quality
            )
            
            if processed_image is None:
                return jsonify({'error': 'Face swap processing failed'}), 500
                
            logger.info("Face swap completed successfully")
            
            # Verify the processed image
            if np.all(processed_image == 0):
                logger.error("Processed image is completely black")
                return jsonify({'error': 'Face swap resulted in black image'}), 500
                
            # Check if face area is black
            face_area = processed_image[int(target_face.bbox[1]):int(target_face.bbox[3]),
                                      int(target_face.bbox[0]):int(target_face.bbox[2])]
            if np.mean(face_area) < 10:  # If average pixel value is very low
                logger.error("Face area is black after processing")
                return jsonify({'error': 'Face area is black after processing'}), 500
            
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