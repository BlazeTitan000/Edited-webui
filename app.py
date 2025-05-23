from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
import cv2
import os
import threading
import numpy as np
from modules.processors.frame.face_swapper import process_frame, pre_check
from modules.face_analyser import get_one_face
import io
from PIL import Image
import onnxruntime as ort
import logging
import time
from datetime import datetime
import base64
from modules.utilities import resolve_relative_path
import insightface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cloud deployment configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure models directory exists and download model if needed
models_dir = resolve_relative_path('models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
pre_check()  # This will download the model if it doesn't exist

# Global variables
uploaded_face_path = None
recording = False
out = None  # Video writer object
source_face = None
execution_provider = None
session = None
FACE_ANALYSER = None
FACE_ANALYSER_MODELS = {}  # Cache for loaded models
last_face_detection = None
last_face_detection_time = 0
FACE_CACHE_DURATION = 0.1  # Reduced cache duration for faster updates
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for consistent output
MIN_FRAME_SIZE = (480, 360)  # Increased minimum frame size for better quality
last_processed_frame = None  # Cache for frame interpolation
processing_enabled = True  # Global flag for processing state
TARGET_FPS = 30  # Target frames per second
FRAME_TIMEOUT = 1.0 / TARGET_FPS  # Maximum time allowed per frame

# Add global variables for frame saving
last_save_time = 0
SAVE_INTERVAL = 3  # seconds
DEBUG_FRAMES_DIR = 'debug_frames'

# Default black rectangle settings
BLACK_RECTANGLE_ENABLED = True
BLACK_RECTANGLE_POSITION = (220, 120)  # Centered position (x, y)
BLACK_RECTANGLE_SIZE = (200, 200)  # Fixed size (width, height)

# Ensure debug frames directory exists
if not os.path.exists(DEBUG_FRAMES_DIR):
    os.makedirs(DEBUG_FRAMES_DIR)

def save_debug_frame(frame, prefix='frame'):
    """Save frame for debugging"""
    global last_save_time
    current_time = time.time()
    
    if current_time - last_save_time >= SAVE_INTERVAL:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(DEBUG_FRAMES_DIR, filename)
        cv2.imwrite(filepath, frame)
        last_save_time = current_time
        logger.info(f"Saved debug frame: {filename}")

def get_onnx_session(provider):
    """Create ONNX session with specified execution provider"""
    try:
        # Use absolute path for the model file
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inswapper_128.onnx')
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        # Create session with maximum performance CUDA settings
        try:
            # Set maximum performance session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.intra_op_num_threads = 4  # Increased threads for faster processing
            session_options.inter_op_num_threads = 4  # Increased threads for faster processing
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Optimized CUDA provider options
            cuda_provider_options = {
                'device_id': 0,
                'gpu_mem_limit': 46 * 1024 * 1024 * 1024,  # Use 46GB of 48GB
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
                'enable_cuda_graph': True
            }
            
            # Create session with CUDA provider only
            session = ort.InferenceSession(
                model_path, 
                providers=[('CUDAExecutionProvider', cuda_provider_options)],
                sess_options=session_options
            )
            logger.info("Successfully created maximum performance CUDA session")
        except Exception as e:
            logger.error(f"Failed to create CUDA session: {e}")
            return None
            
        return session
    except Exception as e:
        logger.error(f"Error creating ONNX session: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return "An error occurred while loading the page. Please check the server logs.", 500

@app.route('/upload_face', methods=['POST'])
def upload_face():
    global uploaded_face_path, source_face, execution_provider, session
    
    if 'face' not in request.files:
        return "No file part", 400
    
    file = request.files['face']
    if file.filename == '':
        return "No selected file", 400
    
    # Force CUDA provider
    execution_provider = 'CUDAExecutionProvider'
    logger.info(f"Using CUDA provider for face processing")
    
    try:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        uploaded_face_path = file_path
        
        # Initialize CUDA session
        start_time = time.time()
        session = get_onnx_session(execution_provider)
        if session is None:
            return "Error initializing face processing with CUDA", 500
        logger.info(f"CUDA session initialization time: {time.time() - start_time:.3f} seconds")
        
        # Pre-process the source face
        start_time = time.time()
        source_face = get_one_face(cv2.imread(uploaded_face_path))
        if source_face is None:
            return "No face detected in the image", 400
        logger.info(f"Face detection time: {time.time() - start_time:.3f} seconds")
            
        logger.info("Face image processed successfully with CUDA")
        return "Success", 200
        
    except Exception as e:
        logger.error(f"Error processing face image: {e}")
        return str(e), 500

def get_face_analyser():
    """Get or create face analyzer with optimized caching"""
    global FACE_ANALYSER, FACE_ANALYSER_MODELS
    
    if FACE_ANALYSER is None:
        try:
            # Initialize face analyzer with CUDA provider
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Prepare with CUDA context and minimal settings
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(160, 160))  # Reduced detection size
            logger.info("Face analyzer initialized with CUDA")
            
            # Cache the models
            FACE_ANALYSER_MODELS = {
                'det_model': FACE_ANALYSER.models.get('detection'),
                'rec_model': FACE_ANALYSER.models.get('recognition'),
                'landmark_model': FACE_ANALYSER.models.get('landmark_2d_106')
            }
        except Exception as e:
            logger.error(f"Error initializing face analyzer: {e}")
            FACE_ANALYSER = None
            return None
    
    return FACE_ANALYSER

def get_one_face_optimized(frame):
    """Ultra-optimized face detection with aggressive caching"""
    global last_face_detection, last_face_detection_time
    
    current_time = time.time()
    if (last_face_detection is not None and 
        current_time - last_face_detection_time < FACE_CACHE_DURATION):
        return last_face_detection
    
    try:
        # Resize frame to minimum size for faster processing
        small_frame = cv2.resize(frame, MIN_FRAME_SIZE)
        
        # Use cached models directly for faster detection
        if FACE_ANALYSER_MODELS:
            det_model = FACE_ANALYSER_MODELS['det_model']
            if det_model:
                faces = det_model.get(small_frame)
                if faces:
                    last_face_detection = min(faces, key=lambda x: x.bbox[0])
                    last_face_detection_time = current_time
                    return last_face_detection
        
        return None
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return None

def blend_frames(current_frame, last_frame, alpha=0.7):
    """Blend frames smoothly to prevent blinking"""
    if last_frame is None or not processing_enabled:
        return current_frame
    return cv2.addWeighted(current_frame, alpha, last_frame, 1 - alpha, 0)

@app.route('/process_frame', methods=['POST'])
def handle_frame():
    global source_face, recording, out, session, frame_counter, last_processed_frame, processing_enabled
    
    if source_face is None:
        return "No face image uploaded. Please upload a face image first.", 400

    if session is None:
        return "Error: Face processing not initialized. Please try uploading the face image again.", 400

    try:
        # Get frame data from request
        frame_data = request.get_data()
        if not frame_data:
            return "No frame data received", 400

        # Convert received data to OpenCV image with minimal processing
        frame = np.array(Image.open(io.BytesIO(frame_data)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Store original frame size
        original_size = frame.shape[:2]
        
        # Resize frame to minimum size for processing
        frame = cv2.resize(frame, MIN_FRAME_SIZE)
        
        # Process the frame with timing
        start_time = time.time()
        
        # Use optimized face detection
        target_face = get_one_face_optimized(frame)
        if target_face:
            try:
                # Process the frame
                processed_frame = process_frame(source_face, frame)
                
                # Resize back to original size with high quality
                processed_frame = cv2.resize(processed_frame, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
                
                # Add black rectangle only when processing is successful
                if BLACK_RECTANGLE_ENABLED:
                    x, y = BLACK_RECTANGLE_POSITION
                    w, h = BLACK_RECTANGLE_SIZE
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
                
                # Cache the processed frame
                last_processed_frame = processed_frame.copy()
                processing_enabled = True
                
            except Exception as e:
                logger.error(f"Error in face processing: {e}")
                # On error, use last processed frame if available
                if last_processed_frame is not None:
                    processed_frame = last_processed_frame.copy()
                else:
                    processed_frame = frame
                processing_enabled = False
        else:
            # No face detected, use last processed frame if available
            if last_processed_frame is not None:
                processed_frame = last_processed_frame.copy()
            else:
                processed_frame = frame
            processing_enabled = False
        
        # Ensure final frame is original size
        processed_frame = cv2.resize(processed_frame, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        processing_time = time.time() - start_time
        logger.info(f"Frame processing time: {processing_time:.3f} seconds ({1/processing_time:.1f} FPS)")

        # Check if we're meeting the target FPS
        if processing_time > FRAME_TIMEOUT:
            logger.warning(f"Frame processing time ({processing_time:.3f}s) exceeds target ({FRAME_TIMEOUT:.3f}s)")

        if recording and out:
            out.write(processed_frame)

        # Convert processed frame to base64 with high quality
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')

        return processed_frame_data, 200

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        # On error, return last processed frame if available
        if last_processed_frame is not None:
            _, buffer = cv2.imencode('.jpg', last_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            processed_frame_data = base64.b64encode(buffer).decode('utf-8')
            return processed_frame_data, 200
        return str(e), 500

@app.route('/record/start', methods=['POST'])
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

@app.route('/record/stop', methods=['POST'])
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

# Pre-load models at startup
def preload_models():
    global session
    logger.info("Pre-loading models...")
    start_time = time.time()
    session = get_onnx_session('CUDAExecutionProvider')
    if session:
        logger.info(f"Models pre-loaded in {time.time() - start_time:.3f} seconds")
    else:
        logger.error("Failed to pre-load models")

# Call preload_models when the application starts
preload_models()

@app.route('/image_swap')
def image_swap():
    return render_template('image_swap.html')

@app.route('/video_swap')
def video_swap():
    return render_template('video_swap.html')

@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    try:
        if 'face' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Missing face or target image'}), 400

        face_file = request.files['face']
        target_file = request.files['target']
        provider = request.form.get('provider', 'CUDAExecutionProvider')

        if face_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read images
        face_image = cv2.imdecode(np.frombuffer(face_file.read(), np.uint8), cv2.IMREAD_COLOR)
        target_image = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Process face swap
        processed_image = process_frame(face_image, target_image, provider)

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': processed_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
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

if __name__ == '__main__':
    # For cloud deployment, use the following:
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Listen on all network interfaces
    app.run(host=host, port=port, debug=False)  # Set debug=False for production
