from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
from app.services.model_service import get_onnx_session, get_one_face_optimized
from app.utils.config import Config
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face

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

        # Initialize session with the selected provider
        session = get_onnx_session(provider)
        if session is None:
            return jsonify({'error': f'Failed to initialize {provider}'}), 500

        # Get source face
        source_face = get_one_face(face_image)
        if source_face is None:
            return jsonify({'error': 'No face detected in source image'}), 400

        # Process face swap
        try:
            processed_image = process_frame(source_face, target_image)
            if processed_image is None:
                return jsonify({'error': 'Face swap processing failed'}), 500
        except Exception as e:
            logger.error(f"Error in face swap processing: {e}")
            return jsonify({'error': f'Face swap processing error: {str(e)}'}), 500

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_base64})

    except Exception as e:
        logger.error(f"Error in face swap: {e}")
        return jsonify({'error': str(e)}), 500 