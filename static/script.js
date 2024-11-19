document.addEventListener('DOMContentLoaded', function() {
    const originalImage = document.getElementById("original-image");
    const resultImage = document.getElementById("result-image");
    const resetButton = document.getElementById("reset-button");
    const statusInfo = document.getElementById("status-info");
    const classSelector = document.getElementById("class-selector");
    let points = [];
    let isAnnotating = true;

    function updateStatusInfo(message, isError = false) {
        if (statusInfo) {
            statusInfo.textContent = message;
            statusInfo.className = isError ? 'error' : 'info';
            statusInfo.style.display = 'block';
        }
    }

    if (originalImage && resultImage) {
        // Create container for points visualization
        const pointsContainer = document.createElement('div');
        pointsContainer.id = 'points-container';
        pointsContainer.style.position = 'absolute';
        pointsContainer.style.top = '0';
        pointsContainer.style.left = '0';
        pointsContainer.style.width = '100%';
        pointsContainer.style.height = '100%';
        originalImage.parentElement.appendChild(pointsContainer);

        // Prevent context menu on right click
        originalImage.addEventListener('contextmenu', function(e) {
            e.preventDefault();
        });

        // Handle mouse clicks for point selection
        originalImage.addEventListener("mousedown", function(event) {
            if (!isAnnotating) return;
            
            event.preventDefault();
            const rect = originalImage.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Calculate scaled coordinates
            const scaleX = originalImage.naturalWidth / rect.width;
            const scaleY = originalImage.naturalHeight / rect.height;
            const scaledX = Math.round(x * scaleX);
            const scaledY = Math.round(y * scaleY);

            // Get selected class from dropdown
            const label = parseInt(classSelector.value, 10);

            // Add point to collection
            points.push({ x: scaledX, y: scaledY, label: label });

            // Visualize point
            const point = document.createElement('div');
            point.className = 'annotation-point';
            point.style.position = 'absolute';
            point.style.left = `${x}px`;
            point.style.top = `${y}px`;
            point.style.width = '10px';
            point.style.height = '10px';
            point.style.marginLeft = '-5px';
            point.style.marginTop = '-5px';
            point.style.borderRadius = '50%';
            point.style.backgroundColor = label === 1 ? '#00ff00' : '#ff0000';
            point.style.border = '2px solid white';
            point.style.boxShadow = '0 0 2px rgba(0,0,0,0.5)';
            pointsContainer.appendChild(point);

            updateStatusInfo(`Added point for class ${label} (${points.length} total). Press Enter when finished.`);
        });

        // Generate masks when Enter is pressed
        document.addEventListener("keydown", function(event) {
            console.log(`Key pressed: ${event.key}`);
            if (event.key === "Enter" && points.length > 0 && isAnnotating) {
                isAnnotating = false; // Stop accepting new points
                console.log("Enter detected, generating segmentation...");
                updateStatusInfo("Generating segmentation...");
                
                fetch("/generate_sam_masks", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ points: points }),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    resultImage.src = data.result_image + "?" + new Date().getTime();
                    updateStatusInfo("Segmentation completed. Press Reset to start over.");
                })
                .catch(error => {
                    console.error("Error:", error);
                    updateStatusInfo("Error generating segmentation", true);
                    isAnnotating = true; // Re-enable annotation on error
                });
            } else if (event.key === "Escape") {
                // Remove last point
                if (points.length > 0 && isAnnotating) {
                    points.pop();
                    if (pointsContainer.lastChild) {
                        pointsContainer.removeChild(pointsContainer.lastChild);
                    }
                    updateStatusInfo(`Removed last point. ${points.length} points remaining.`);
                }
            }
        });

        // Reset functionality
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                fetch("/reset_segmentation", {
                    method: "POST"
                })
                .then(response => response.json())
                .then(data => {
                    resultImage.src = data.result_image + "?" + new Date().getTime();
                    points = []; // Reset points array
                    // Clear visualization points
                    while (pointsContainer.firstChild) {
                        pointsContainer.removeChild(pointsContainer.firstChild);
                    }
                    isAnnotating = true; // Re-enable annotation
                    updateStatusInfo("Reset complete. Add points and press Enter when finished.");
                })
                .catch(error => {
                    console.error("Error:", error);
                    updateStatusInfo("Error resetting segmentation", true);
                });
            });
        }

        // Add helpful tooltips
        const tooltipDiv = document.createElement('div');
        tooltipDiv.className = 'tooltip';
        tooltipDiv.innerHTML = `
            <p><strong>Controls:</strong></p>
            <p>ESC: Remove last point</p>
            <p>Enter: Generate segmentation</p>
        `;
        document.body.appendChild(tooltipDiv);
    }

    // Add some basic styles
    const style = document.createElement('style');
    style.textContent = `
        .tooltip {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            z-index: 1000;
        }
        .info {
            color: #2196F3;
        }
        .error {
            color: #f44336;
        }
        .annotation-point {
            transition: transform 0.1s ease-out;
        }
        .annotation-point:hover {
            transform: scale(1.5);
        }
    `;
    document.head.appendChild(style);
});

        
document.addEventListener('DOMContentLoaded', function() {
    const yoloUploadBtn = document.getElementById('yolo-upload-btn');
    const yoloImageInput = document.getElementById('yolo-image-input');
    const yoloSubmitMessage = document.getElementById('yolo-submit-message');
    const yoloResultContainer = document.getElementById('yolo-result-container');
    const yoloOriginalImage = document.getElementById('yolo-original-image');
    const yoloResultImage = document.getElementById('yolo-result-image');
    const detectionList = document.getElementById('detection-list');

    if (yoloUploadBtn && yoloImageInput) {
        yoloUploadBtn.addEventListener('click', () => yoloImageInput.click());

        yoloImageInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const formData = new FormData();
                formData.append('image', file);

                // Upload the image
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) throw new Error('Upload failed');
                    return response.text();
                })
                .then(() => {
                    // Show upload success message
                    yoloSubmitMessage.style.display = 'block';
                    yoloSubmitMessage.textContent = 'Image uploaded successfully';

                    // Display original image
                    yoloOriginalImage.src = URL.createObjectURL(file);
                    yoloResultContainer.style.display = 'block';

                    // Run YOLO prediction
                    return fetch('/predict_yolo', {
                        method: 'POST'
                    });
                })
                .then(response => response.json())
                .then(data => {
                    // Display prediction result
                    yoloResultImage.src = data.result_image + "?" + new Date().getTime();

                    // Display detection information
                    detectionList.innerHTML = '';
                    data.detections.forEach((detection, index) => {
                        const detectionDiv = document.createElement('div');
                        detectionDiv.className = 'detection-item';
                        detectionDiv.innerHTML = `
                            <p>Detection ${index + 1}:</p>
                            <ul>
                                <li>Confidence: ${(detection.confidence * 100).toFixed(1)}%</li>
                                <li>Class: ${detection.class}</li>
                                <li>Location: (${detection.box.map(v => Math.round(v)).join(', ')})</li>
                            </ul>
                        `;
                        detectionList.appendChild(detectionDiv);
                    });
                })
                .catch(error => {
                    console.error('Error:', error);
                    yoloSubmitMessage.style.display = 'block';
                    yoloSubmitMessage.textContent = 'Error processing image';
                });
            }
        });
    }
});


document.addEventListener('DOMContentLoaded', function() {
    const retrainButton = document.getElementById('save-retrain-button');
    if (retrainButton) {
        retrainButton.addEventListener('click', function() {
            // Vérifier si une segmentation a été générée
            const resultImage = document.getElementById('result-image');
            if (!resultImage || !resultImage.src.includes('current_result.jpg')) {
                alert("Please generate segmentation first");
                return;
            }

            // Désactiver le bouton et montrer l'état
            retrainButton.disabled = true;
            retrainButton.textContent = 'Saving & Training...';
            
            // Créer un élément pour afficher la progression
            let progressDiv = document.getElementById('training-progress');
            if (!progressDiv) {
                progressDiv = document.createElement('div');
                progressDiv.id = 'training-progress';
                progressDiv.style.marginTop = '10px';
                retrainButton.parentNode.appendChild(progressDiv);
            }
            progressDiv.textContent = 'Saving training data...';

            fetch("/save_for_training", {
                method: "POST",
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                progressDiv.textContent = `Training complete!`;
                retrainButton.disabled = false;
                retrainButton.textContent = 'Save & Retrain';
            })
            .catch(error => {
                console.error("Error:", error);
                progressDiv.textContent = `Error: ${error.message}`;
                progressDiv.style.color = 'red';
                retrainButton.disabled = false;
                retrainButton.textContent = 'Save & Retrain';
            });
        });
    }
});