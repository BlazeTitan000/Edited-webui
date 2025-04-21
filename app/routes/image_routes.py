from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face, get_face_analyser
import insightface

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

        # Initialize face analyzer
        face_analyzer = get_face_analyser()
        if face_analyzer is None:
            return jsonify({'error': 'Failed to initialize face analyzer'}), 500

        # Get source face with detailed logging
        logger.info("Detecting face in source image...")
        source_face = face_analyzer.get(face_image)
        if not source_face:
            return jsonify({'error': 'No face detected in source image'}), 400
        source_face = min(source_face, key=lambda x: x.bbox[0])
        logger.info(f"Source face detected with bbox: {source_face.bbox}")

        # Get target face with detailed logging
        logger.info("Detecting face in target image...")
        target_face = face_analyzer.get(target_image)
        if not target_face:
            return jsonify({'error': 'No face detected in target image'}), 400
        target_face = min(target_face, key=lambda x: x.bbox[0])
        logger.info(f"Target face detected with bbox: {target_face.bbox}")

        # Initialize session with the selected provider
        session = get_onnx_session(provider)
        if session is None:
            return jsonify({'error': f'Failed to initialize {provider}'}), 500

        # Process face swap with detailed logging
        logger.info("Starting face swap processing...")
        try:
            # Get face swapper model
            model_path = Config.MODEL_PATH
            face_swapper = insightface.model_zoo.get_model(model_path, providers=[provider])
            
            # Perform face swap
            processed_image = face_swapper.get(target_image, target_face, source_face, paste_back=True)
            
            if processed_image is None:
                return jsonify({'error': 'Face swap processing failed'}), 500
                
            logger.info("Face swap completed successfully")
            
        except Exception as e:
            logger.error(f"Error in face swap processing: {str(e)}")
            return jsonify({'error': f'Face swap processing error: {str(e)}'}), 500

        # Convert to base64 with high quality
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_base64})

    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return jsonify({'error': str(e)}), 500 