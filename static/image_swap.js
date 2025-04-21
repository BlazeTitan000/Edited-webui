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

    // Validate file types
    if (!faceFile.type.startsWith('image/') || !targetFile.type.startsWith('image/')) {
        document.getElementById('error-message').innerText = "Please select valid image files.";
        return;
    }

    const formData = new FormData();
    formData.append('face', faceFile);
    formData.append('target', targetFile);
    formData.append('providers', provider);

    // Disable the button during processing
    const swapButton = document.getElementById('swap-button');
    swapButton.disabled = true;
    swapButton.innerText = "Processing...";

    document.getElementById('progress-bar').innerText = "Processing...";
    document.getElementById('error-message').innerText = "";
    document.getElementById('message').innerText = "";

    try {
        const response = await fetch(swapUrl, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            if (result.processed_image) {
                document.getElementById('result-preview').src = 'data:image/jpeg;base64,' + result.processed_image;
                document.getElementById('progress-bar').innerText = "Processing complete!";
                document.getElementById('message').innerText = "Face swap successful!";
            } else {
                throw new Error("No processed image received from server");
            }
        } else {
            const error = await response.text();
            throw new Error(error || "Server error occurred");
        }
    } catch (error) {
        console.error("Face swap error:", error);
        document.getElementById('error-message').innerText = "Error: " + error.message;
        document.getElementById('progress-bar').innerText = "Error occurred";
    } finally {
        // Re-enable the button
        swapButton.disabled = false;
        swapButton.innerText = "Swap Faces";
    }
}); 