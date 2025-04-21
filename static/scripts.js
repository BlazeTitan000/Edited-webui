let isRecording = false;
let ws = null;
let mediaStream = null;
let canvas = document.getElementById('webcam-canvas');
let ctx = canvas.getContext('2d');
let video = document.getElementById('webcam');
let liveStream = document.getElementById('live-stream');

// Initialize webcam
async function initWebcam() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        video.srcObject = mediaStream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
        return true;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        document.getElementById('error-message').innerText = "Error accessing webcam. Please ensure you have granted camera permissions.";
        return false;
    }
}

function uploadFace() {
    const fileInput = document.getElementById('face');
    const file = fileInput.files[0];

    if (!file) {
        document.getElementById('error-message').innerText = "Please select a file first.";
        return;
    }

    // Validate file type
    if (!file.type.match('image.*')) {
        document.getElementById('error-message').innerText = "Please select an image file.";
        return;
    }

    // Get selected execution provider
    const providerSelect = document.getElementById('providers');
    const selectedProvider = providerSelect.value;

    var formData = new FormData();
    formData.append('face', file);
    formData.append('provider', selectedProvider);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', uploadUrl, true);

    xhr.onload = function () {
        if (xhr.status === 200) {
            document.getElementById('live-button').disabled = false;
            document.getElementById('record-button').disabled = false;

            var messageElement = document.getElementById('message');
            messageElement.innerText = "File uploaded successfully!";
            document.getElementById('error-message').innerText = "";

            setTimeout(function () {
                messageElement.innerText = "";
            }, 5000);

        } else {
            document.getElementById('message').innerText = "";
            document.getElementById('error-message').innerText = "An error occurred during file upload: " + xhr.statusText;
        }
    };

    xhr.onerror = function () {
        document.getElementById('error-message').innerText = "Network error occurred during upload.";
    };

    xhr.send(formData);
}

async function startLive() {
    if (!mediaStream) {
        const success = await initWebcam();
        if (!success) return;
    }

    // Connect to WebSocket
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("WebSocket connection established");
        startStreaming();
        document.getElementById('progress-bar').innerText = "Connected and streaming...";
    };

    ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
            const url = URL.createObjectURL(event.data);
            liveStream.src = url;
            URL.revokeObjectURL(url);
        } else {
            // Handle text messages (errors)
            document.getElementById('error-message').innerText = event.data;
        }
    };

    ws.onclose = () => {
        console.log("WebSocket connection closed");
        stopStreaming();
        document.getElementById('progress-bar').innerText = "Connection closed";
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        document.getElementById('error-message').innerText = "WebSocket connection error";
    };

    document.getElementById('stop-button').disabled = false;
    document.getElementById('live-button').disabled = true;
}

function startStreaming() {
    function captureFrame() {
        if (!mediaStream || !ws || ws.readyState !== WebSocket.OPEN) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
            ws.send(blob);
        }, 'image/jpeg', 0.8);

        requestAnimationFrame(captureFrame);
    }

    captureFrame();
}

function stopStreaming() {
    if (ws) {
        ws.close();
        ws = null;
    }

    document.getElementById('stop-button').disabled = true;
    document.getElementById('live-button').disabled = false;
    liveStream.src = '';
}

function stopLive() {
    stopStreaming();

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Stop recording if it's active
    if (isRecording) {
        stopRecording();
    }
}

function startRecording() {
    isRecording = true;
    document.getElementById('record-button').disabled = true;
    document.getElementById('stop-record-button').disabled = false;

    // Send a request to the server to start recording
    var xhr = new XMLHttpRequest();
    xhr.open('POST', startRecordUrl, true);

    xhr.onload = function () {
        if (xhr.status === 204) {
            document.getElementById('message').innerText = "Recording started";
        } else {
            document.getElementById('error-message').innerText = "Failed to start recording";
        }
    };

    xhr.onerror = function () {
        document.getElementById('error-message').innerText = "Network error while starting recording";
    };

    xhr.send();
}

function stopRecording() {
    isRecording = false;
    document.getElementById('record-button').disabled = false;
    document.getElementById('stop-record-button').disabled = true;

    // Send a request to the server to stop recording
    var xhr = new XMLHttpRequest();
    xhr.open('POST', stopRecordUrl, true);

    xhr.onload = function () {
        if (xhr.status === 204) {
            document.getElementById('message').innerText = "Recording stopped";
        } else {
            document.getElementById('error-message').innerText = "Failed to stop recording";
        }
    };

    xhr.onerror = function () {
        document.getElementById('error-message').innerText = "Network error while stopping recording";
    };

    xhr.send();
}
