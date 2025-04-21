// Preview images and videos when selected
document.getElementById('face').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById('source-preview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('video').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        document.getElementById('target-preview').src = url;
    }
});

// Handle video processing
document.getElementById('process-button').addEventListener('click', async function () {
    const faceFile = document.getElementById('face').files[0];
    const videoFile = document.getElementById('video').files[0];
    const provider = document.getElementById('providers').value;

    if (!faceFile || !videoFile) {
        document.getElementById('error-message').innerText = "Please select both source face and target video.";
        return;
    }

    const formData = new FormData();
    formData.append('face', faceFile);
    formData.append('video', videoFile);
    formData.append('provider', provider);

    document.getElementById('progress-bar').innerText = "Processing video...";
    document.getElementById('error-message').innerText = "";

    try {
        const response = await fetch(processVideoUrl, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            document.getElementById('result-preview').src = url;
            document.getElementById('progress-bar').innerText = "Processing complete!";
            document.getElementById('message').innerText = "Video processing successful!";
        } else {
            const error = await response.text();
            document.getElementById('error-message').innerText = "Error: " + error;
            document.getElementById('progress-bar').innerText = "Error occurred";
        }
    } catch (error) {
        document.getElementById('error-message').innerText = "Network error: " + error.message;
        document.getElementById('progress-bar').innerText = "Error occurred";
    }
}); 