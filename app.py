from flask import Flask, render_template, request, redirect, url_for, Response
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

# Add global variables for frame saving
last_save_time = 0
SAVE_INTERVAL = 3  # seconds
DEBUG_FRAMES_DIR = 'debug_frames'

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
        
        # Create session with maximum performance settings
        try:
            # Set maximum performance session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.intra_op_num_threads = 16  # Maximum threads for parallelization
            session_options.inter_op_num_threads = 16  # Maximum threads for parallelization
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.add_session_config_entry('session.load_model_format', 'ONNX')
            session_options.add_session_config_entry('session.use_deterministic_compute', '0')
            
            # Maximum performance CUDA options
            cuda_provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 45 * 1024 * 1024 * 1024,  # Use 45GB of 48GB
                'cudnn_conv_algo_search': 'HEURISTIC',  # Faster than EXHAUSTIVE
                'do_copy_in_default_stream': True,
                'enable_cuda_graph': True,
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv_algo': '0',
                'enable_tensorrt': '1',  # Enable TensorRT optimization
                'tensorrt_fp16_enable': '1',  # Enable FP16 precision
                'tensorrt_max_workspace_size': 2147483648,  # 2GB workspace
                'tensorrt_max_partition_iterations': 10,
                'tensorrt_min_subgraph_size': 1,
                'tensorrt_dump_subgraphs': '0',
                'tensorrt_engine_cache_enable': '1',
                'tensorrt_engine_cache_path': './trt_cache',
            }
            
            # Create session with maximum performance settings
            session = ort.InferenceSession(
                model_path, 
                providers=[('CUDAExecutionProvider', cuda_provider_options)],
                sess_options=session_options
            )
            logger.info("Successfully created maximum performance ONNX session")
        except Exception as e:
            logger.error(f"Failed to create maximum performance session: {e}")
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
    
    # Force CUDA provider for RTX A6000
    execution_provider = 'CUDAExecutionProvider'
    logger.info(f"Using CUDA provider for RTX A6000")
    
    try:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        uploaded_face_path = file_path
        
        # Initialize optimized ONNX session
        session = get_onnx_session(execution_provider)
        if session is None:
            return "Error initializing face processing with CUDA", 500
        
        # Pre-process the source face
        source_face = get_one_face(cv2.imread(uploaded_face_path))
        if source_face is None:
            return "No face detected in the image", 400
            
        logger.info("Face image processed successfully with CUDA")
        return "Success", 200
        
    except Exception as e:
        logger.error(f"Error processing face image: {e}")
        return str(e), 500

@app.route('/process_frame', methods=['POST'])
def handle_frame():
    global source_face, recording, out, session
    
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
        
        # Optional: Resize frame for faster processing (uncomment if needed)
        # frame = cv2.resize(frame, (640, 480))  # Adjust size as needed

        # Process the frame with timing
        start_time = time.time()
        processed_frame = process_frame(source_face, frame)
        processing_time = time.time() - start_time
        logger.info(f"Frame processing time: {processing_time:.3f} seconds ({1/processing_time:.1f} FPS)")

        if recording and out:
            out.write(processed_frame)

        # Convert processed frame to base64 with minimal quality loss
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')

        return processed_frame_data, 200

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
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

if __name__ == '__main__':
    # For cloud deployment, use the following:
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Listen on all network interfaces
    app.run(host=host, port=port, debug=False)  # Set debug=False for production
