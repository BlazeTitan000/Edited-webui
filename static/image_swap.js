// Preview images when selected
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

document.getElementById('target').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById('target-preview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Handle face swap
document.getElementById('swap-button').addEventListener('click', async function () {
    const faceFile = document.getElementById('face').files[0];
    const targetFile = document.getElementById('target').files[0];
    const provider = document.getElementById('providers').value;

    if (!faceFile || !targetFile) {
        document.getElementById('error-message').innerText = "Please select both source face and target images.";
        return;
    }

    const formData = new FormData();
    formData.append('face', faceFile);
    formData.append('target', targetFile);
    formData.append('provider', provider);

    document.getElementById('progress-bar').innerText = "Processing...";
    document.getElementById('error-message').innerText = "";

    try {
        const response = await fetch(swapUrl, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('result-preview').src = 'data:image/jpeg;base64,' + result.processed_image;
            document.getElementById('progress-bar').innerText = "Processing complete!";
            document.getElementById('message').innerText = "Face swap successful!";
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