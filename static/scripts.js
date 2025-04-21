// Define global configuration
const config = {
    frameInterval: 1000 / 30, // 30 FPS (33.33ms between frames)
    maxErrors: 5,      // maximum number of errors before stopping
    jpegQuality: 0.8,  // JPEG compression quality (0-1)
    targetFPS: 30      // Target frames per second
};

let isRecording = false;
let mediaStream = null;
let canvas = document.getElementById('webcam-canvas');
let ctx = canvas.getContext('2d');
let video = document.getElementById('webcam');
let liveStream = document.getElementById('live-stream');
let processingInterval = null;
let errorCount = 0;
let lastFrameTime = 0;
let frameCounter = 0;
let fpsCounter = 0;
let uploadCounter = 0;
let fpsTimer = null;
let lastUploadTime = 0;
let uploadSize = 0;

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateStatus() {
    const currentTime = performance.now();
    const uploadRate = uploadSize / ((currentTime - lastUploadTime) / 1000); // bytes per second
    document.getElementById('progress-bar').innerText =
        `Processing: ${fpsCounter} FPS | Upload: ${formatBytes(uploadRate)}/s`;
    fpsCounter = 0;
    uploadSize = 0;
}

// Initialize webcam
async function initWebcam() {
    try {
        console.log("Initializing webcam...");
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                frameRate: 30
            }
        });
        console.log("Webcam access granted");
        video.srcObject = mediaStream;
        video.onloadedmetadata = () => {
            console.log(`Webcam resolution: ${video.videoWidth}x${video.videoHeight}`);
            // Set fixed dimensions for canvas and video
            canvas.width = 640;
            canvas.height = 480;
            video.width = 640;
            video.height = 480;
        };
        return true;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        document.getElementById('error-message').innerText = "Error accessing webcam. Please ensure you have granted camera permissions.";
        return false;
    }
}

function uploadFace() {
    console.log("Starting face upload...");
    const fileInput = document.getElementById('face');
    const file = fileInput.files[0];

    if (!file) {
        console.error("No file selected");
        document.getElementById('error-message').innerText = "Please select a file first.";
        return;
    }

    // Validate file type
    if (!file.type.match('image.*')) {
        console.error("Invalid file type:", file.type);
        document.getElementById('error-message').innerText = "Please select an image file.";
        return;
    }

    // Get selected execution provider
    const providerSelect = document.getElementById('providers');
    const selectedProvider = providerSelect.value;
    console.log("Selected provider:", selectedProvider);

    var formData = new FormData();
    formData.append('face', file);
    formData.append('provider', selectedProvider);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', uploadUrl, true);

    xhr.onload = function () {
        console.log("Upload response status:", xhr.status);
        if (xhr.status === 200) {
            document.getElementById('live-button').disabled = false;
            document.getElementById('record-button').disabled = false;

            var messageElement = document.getElementById('message');
            messageElement.innerText = "Face image processed successfully!";
            document.getElementById('error-message').innerText = "";
            document.getElementById('progress-bar').innerText = "Ready to go live";
            console.log("Face upload successful");

            setTimeout(function () {
                messageElement.innerText = "";
            }, 5000);

        } else {
            console.error("Upload failed:", xhr.statusText);
            document.getElementById('message').innerText = "";
            document.getElementById('error-message').innerText = "An error occurred during face processing: " + xhr.statusText;
            document.getElementById('progress-bar').innerText = "Face processing failed";
        }
    };

    xhr.onerror = function () {
        console.error("Network error during upload");
        document.getElementById('error-message').innerText = "Network error occurred during upload.";
        document.getElementById('progress-bar').innerText = "Upload failed";
    };

    xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            console.log(`Upload progress: ${percentComplete}%`);
            if (percentComplete < 100) {
                document.getElementById('progress-bar').innerText = `Uploading face image: ${Math.round(percentComplete)}%`;
            } else {
                document.getElementById('progress-bar').innerText = "Processing face image...";
            }
        }
    };

    xhr.send(formData);
}

async function startLive() {
    if (!mediaStream) {
        const success = await initWebcam();
        if (!success) return;
    }

    // Start status update timer
    fpsTimer = setInterval(updateStatus, 1000);
    lastUploadTime = performance.now();

    // Start processing frames
    processingInterval = setInterval(processFrame, config.frameInterval);
    document.getElementById('progress-bar').innerText = "Connected and streaming...";
    document.getElementById('stop-button').disabled = false;
    document.getElementById('live-button').disabled = true;
}

async function processFrame() {
    if (!mediaStream) {
        console.warn("No media stream available");
        return;
    }

    const currentTime = performance.now();
    const elapsed = currentTime - lastFrameTime;

    // Skip frame if we're running behind
    if (elapsed < config.frameInterval) {
        return;
    }

    try {
        const startTime = performance.now();

        // Capture frame from canvas with fixed dimensions
        ctx.drawImage(video, 0, 0, 640, 480);
        const frameBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', config.jpegQuality));

        // Track upload size
        uploadSize += frameBlob.size;

        // Send frame to server
        const response = await fetch('/process_frame', {
            method: 'POST',
            body: frameBlob
        });

        if (response.ok) {
            const processedFrameData = await response.text();
            liveStream.src = 'data:image/jpeg;base64,' + processedFrameData;
            // Ensure live stream maintains fixed dimensions
            liveStream.width = 640;
            liveStream.height = 480;

            const endTime = performance.now();
            const processingTime = endTime - startTime;
            fpsCounter++;

            // Log performance metrics
            if (processingTime > config.frameInterval) {
                console.warn(`Frame processing time (${processingTime.toFixed(2)}ms) exceeds target (${config.frameInterval}ms)`);
            }
        } else {
            const error = await response.text();
            console.error("Server error:", error);
            document.getElementById('error-message').innerText = `Error processing frame: ${error}`;

            errorCount = (errorCount || 0) + 1;
            if (errorCount > config.maxErrors) {
                console.error("Too many errors, stopping processing");
                stopLive();
            }
        }
    } catch (error) {
        console.error("Error processing frame:", error);
        document.getElementById('error-message').innerText = "Error processing frame. Please try again.";

        errorCount = (errorCount || 0) + 1;
        if (errorCount > config.maxErrors) {
            console.error("Too many errors, stopping processing");
            stopLive();
        }
    }

    lastFrameTime = currentTime;
}

function stopLive() {
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }

    if (fpsTimer) {
        clearInterval(fpsTimer);
        fpsTimer = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Stop recording if it's active
    if (isRecording) {
        stopRecording();
    }

    document.getElementById('stop-button').disabled = true;
    document.getElementById('live-button').disabled = false;
    liveStream.src = '';
    document.getElementById('progress-bar').innerText = "Streaming stopped";
}

function startRecording() {
    console.log("Starting recording...");
    isRecording = true;
    document.getElementById('record-button').disabled = true;
    document.getElementById('stop-record-button').disabled = false;

    // Send a request to the server to start recording
    var xhr = new XMLHttpRequest();
    xhr.open('POST', startRecordUrl, true);

    xhr.onload = function () {
        console.log("Start recording response:", xhr.status);
        if (xhr.status === 204) {
            document.getElementById('message').innerText = "Recording started";
            console.log("Recording started successfully");
        } else {
            console.error("Failed to start recording:", xhr.statusText);
            document.getElementById('error-message').innerText = "Failed to start recording";
            isRecording = false;
            document.getElementById('record-button').disabled = false;
            document.getElementById('stop-record-button').disabled = true;
        }
    };

    xhr.onerror = function () {
        console.error("Network error while starting recording");
        document.getElementById('error-message').innerText = "Network error while starting recording";
        isRecording = false;
        document.getElementById('record-button').disabled = false;
        document.getElementById('stop-record-button').disabled = true;
    };

    xhr.send();
}

function stopRecording() {
    console.log("Stopping recording...");
    isRecording = false;
    document.getElementById('record-button').disabled = false;
    document.getElementById('stop-record-button').disabled = true;

    // Send a request to the server to stop recording
    var xhr = new XMLHttpRequest();
    xhr.open('POST', stopRecordUrl, true);

    xhr.onload = function () {
        console.log("Stop recording response:", xhr.status);
        if (xhr.status === 204) {
            document.getElementById('message').innerText = "Recording stopped";
            console.log("Recording stopped successfully");
        } else {
            console.error("Failed to stop recording:", xhr.statusText);
            document.getElementById('error-message').innerText = "Failed to stop recording";
        }
    };

    xhr.onerror = function () {
        console.error("Network error while stopping recording");
        document.getElementById('error-message').innerText = "Network error while stopping recording";
    };

    xhr.send();
}
