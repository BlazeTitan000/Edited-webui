import onnxruntime as ort
import logging
import insightface
from app.utils.config import Config
import os
import time
import cv2

logger = logging.getLogger(__name__)

# Global variables
session = None
FACE_ANALYSER = None
FACE_ANALYSER_MODELS = {}
last_face_detection = None
last_face_detection_time = 0

def get_onnx_session(provider='CUDAExecutionProvider'):
    """Create ONNX session with specified execution provider"""
    try:
        logger.info(f"Attempting to load model from: {Config.MODEL_PATH}")
        
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(f"Model file not found at: {Config.MODEL_PATH}")
            return None
            
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 4
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            cuda_provider_options = {
                'device_id': 0,
                'gpu_mem_limit': 46 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
                'enable_cuda_graph': True
            }
            
            session = ort.InferenceSession(
                Config.MODEL_PATH, 
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
        return None

def get_face_analyser():
    """Get or create face analyzer with optimized caching"""
    global FACE_ANALYSER, FACE_ANALYSER_MODELS
    
    if FACE_ANALYSER is None:
        try:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(160, 160))
            logger.info("Face analyzer initialized with CUDA")
            
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
        current_time - last_face_detection_time < Config.FACE_CACHE_DURATION):
        return last_face_detection
    
    try:
        small_frame = cv2.resize(frame, Config.MIN_FRAME_SIZE)
        
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

def preload_models():
    """Preload models at application startup"""
    global session
    logger.info("Pre-loading models...")
    start_time = time.time()
    session = get_onnx_session('CUDAExecutionProvider')
    if session:
        logger.info(f"Models pre-loaded in {time.time() - start_time:.3f} seconds")
    else:
        logger.error("Failed to pre-load models") 