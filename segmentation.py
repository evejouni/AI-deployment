# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
import torch
import sys
import traceback
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import json
import shutil
from datetime import datetime

app = Flask(__name__)

# Configure folders
BASE_UPLOAD_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'images')
SAM_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'sam_output')
YOLO_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'yolo_output')
TRAIN_DATA_FOLDER = 'data/train'
TRAIN_IMAGES_FOLDER = os.path.join(TRAIN_DATA_FOLDER, 'images')
TRAIN_LABELS_FOLDER = os.path.join(TRAIN_DATA_FOLDER, 'labels')
DATA_FOLDER = 'data'

# Configure Flask paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['YOLO_FOLDER'] = YOLO_FOLDER
app.config['SAM_FOLDER'] = SAM_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(YOLO_FOLDER, exist_ok=True)
os.makedirs(SAM_FOLDER, exist_ok=True)
os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TRAIN_LABELS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Global variables
class GlobalVars:
    current_image = None
    current_mask_combined = None
    uploaded_image_path = ""
    original_filename = ""
    predictor = None
    sam = None
    sam_masks = None
    yolo_model = None

g = GlobalVars()

def load_sam_model():
    """Load SAM model with proper error handling"""
    try:
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        if not os.path.exists(sam_checkpoint):
            print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
            return None, None

        print("Loading SAM model...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        predictor = SamPredictor(sam)
        print("SAM model loaded successfully")
        return sam, predictor
    except Exception as e:
        print(f"Error loading SAM model: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return None, None

def load_yolo_model():
    """Load YOLO model with proper error handling"""
    try:
        model_path = "best.pt"  
        if not os.path.exists(model_path):
            print(f"Error: YOLO model not found at {model_path}")
            return None
            
        print("Loading YOLO model...")
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return None
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def clear_folder(folder_path):
    """Clear all files in a folder"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def mask_to_yolo_polygon(mask, class_id):
    """
    Convert a binary mask to YOLO polygon format.
    Args:
        mask: Binary mask array
        class_id: Class ID for the annotation (0 for puces, 1 for bulles)
    Returns:
        List of strings, each string is a YOLO format polygon annotation
    """
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    annotations = []
    for contour in contours:
        # Simplifier le contour pour réduire le nombre de points
        epsilon = 0.0005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convertir en format YOLO
        if len(approx) >= 3:  # Vérifier qu'on a au moins 3 points pour former un polygone
            yolo_line = f"{class_id}"
            for point in approx[:, 0, :]:
                x_norm = point[0] / width
                y_norm = point[1] / height
                yolo_line += f" {x_norm:.6f} {y_norm:.6f}"
            annotations.append(yolo_line)
    
    return annotations


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        try:
            # Clear existing files in upload folder
            clear_folder(app.config['UPLOAD_FOLDER'])

            g.original_filename = "current_image.jpg" 
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], g.original_filename)
            file.save(file_path)
            g.uploaded_image_path = file_path
            
            # Load image and prepare for SAM
            g.current_image = cv2.imread(file_path)
            if g.current_image is None:
                return "Failed to load image", 500
            
            # Initialize SAM predictor with the image
            print("initializing SAM")
            if g.predictor is not None:
                image_rgb = cv2.cvtColor(g.current_image, cv2.COLOR_BGR2RGB)
                g.predictor.set_image(image_rgb)
                print("Image set in SAM predictor")
            
            return "File uploaded successfully", 200
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return f"Error during upload: {str(e)}", 500
    
    return "File type not allowed", 400



@app.route('/generate_sam_masks', methods=['POST'])
def generate_sam_masks():
    try:
        if g.predictor is None:
            return jsonify({"error": "SAM model not loaded"}), 500
        
        if g.current_image is None:
            return jsonify({"error": "No image loaded"}), 400

        # Get points from request
        data = request.json
        points = data.get("points", [])
        
        if not points:
            return jsonify({"error": "No points provided"}), 400

        # Prepare input points and labels for SAM
        input_points = np.array([[p['x'], p['y']] for p in points])
        input_labels = np.array([p['label'] for p in points])

        # Convert labels for SAM (foreground/background binary format)
        puces_points = []
        puces_labels = []
        bulles_points = []
        bulles_labels = []  # Store original class labels
        
        for i, label in enumerate(input_labels):
            if label == 1:  # Puces
                puces_points.append(input_points[i])
                puces_labels.append(1)
            elif label == 2:  # Bulles
                bulles_points.append(input_points[i])
                bulles_labels.append(1)

        # Initialize masks lists for each class
        puces_masks = []
        if puces_points:
            puces_points = np.array(puces_points)
            puces_labels = np.array(puces_labels)
            masks_puces, _, _ = g.predictor.predict(
                point_coords=puces_points,
                point_labels=puces_labels,
                multimask_output=True
            )
            puces_masks = [masks_puces[0]]

        bulles_masks = []
        if bulles_points:
            bulles_points = np.array(bulles_points)
            bulles_labels = np.array(bulles_labels)
            masks_bulles, _, _ = g.predictor.predict(
                point_coords=bulles_points,
                point_labels=bulles_labels,
                multimask_output=True
            )
            bulles_masks = [masks_bulles[0]]

        # Debug info
        print(f"Nombre de points puces: {len(puces_points)}")
        print(f"Nombre de points bulles: {len(bulles_points)}")
        print(f"Forme du masque puces: {[m.shape for m in puces_masks] if puces_masks else 'Aucun'}")
        print(f"Forme du masque bulles: {[m.shape for m in bulles_masks] if bulles_masks else 'Aucun'}")

        class_colors = [
            (0, 255, 0),    
        ]

        if puces_masks:
            puces_mask = puces_masks[0].astype(bool).astype(np.uint8)
        else:
            puces_mask = np.zeros(g.current_image.shape[:2], dtype=np.uint8)

        if bulles_masks:
            bulles_mask = bulles_masks[0].astype(bool).astype(np.uint8)
        else:
            bulles_mask = np.zeros(g.current_image.shape[:2], dtype=np.uint8)
            
        # Vérifier les valeurs des masques
        print("Vérification des masques:")
        print(f"Valeurs uniques masque puces: {np.unique(puces_mask)}")
        print(f"Valeurs uniques masque bulles: {np.unique(bulles_mask)}")

        # Create visualization
        class_colors = [(0, 255, 0), (255, 0, 0)]  # Vert pour "puces", rouge pour "bulles"
        combined_masks = puces_masks + bulles_masks

        combined_overlay = create_sam_visualization(
            g.current_image, 
            [puces_mask, bulles_mask], 
            class_colors
        )

        # Save result
        result_path = os.path.join(app.config['SAM_FOLDER'], 'current_result.jpg')
        if os.path.exists(result_path):
            os.remove(result_path)
        cv2.imwrite(result_path, combined_overlay)

        return jsonify({
            "message": "SAM segmentation completed",
            "result_image": os.path.join('static', 'sam_output', 'current_result.jpg')
        })

    except Exception as e:
        print(f"SAM segmentation error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



def create_sam_visualization(image, masks, class_colors, alpha=0.5):
    """
    Create visualization of SAM masks with improved overlay.
    
    Args:
        image: Input image (numpy array)
        masks: List of binary masks (one per class)
        class_colors: List of tuples representing RGB colors for each class
        alpha: Transparency factor for blending masks
        
    Returns:
        Visualization image with properly overlaid masks
    """
    # Convert image to float32 for processing
    result = image.copy().astype(np.float32)
    
    # Create a combined overlay for all masks
    overlay = np.zeros_like(result)
    
    # Create a combined alpha channel
    alpha_channel = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Process each mask separately
    for mask, color in zip(masks, class_colors):
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        # Convert to float32 for processing
        mask = mask.astype(np.float32)
        
        # Create colored mask
        colored_mask = np.zeros_like(result)
        for i in range(3):
            colored_mask[..., i] = mask * color[i]
            
        # Update overlay and alpha channel
        mask_alpha = mask * alpha
        alpha_channel = np.maximum(alpha_channel, mask_alpha)
        overlay = np.where(np.expand_dims(mask, -1) > 0, colored_mask, overlay)
    
    # Combine the original image with the overlay using the alpha channel
    alpha_3d = np.expand_dims(alpha_channel, -1)
    result = (1 - alpha_3d) * result + alpha_3d * overlay
    
    # Ensure the output is in the valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


@app.route('/save_for_training', methods=['POST'])
def save_for_training():
    print("Starting save and retrain process...")
    try:
        if g.current_image is None:
            return jsonify({"error": "No image or masks available"}), 400
        
        # Obtenir les masques depuis le résultat SAM
        result_path = os.path.join(app.config['SAM_FOLDER'], 'current_result.jpg')
        if not os.path.exists(result_path):
            return jsonify({"error": "No segmentation result available"}), 400
            
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"training_sample_{timestamp}"
        
        # Sauvegarder l'image
        image_path = os.path.join(TRAIN_IMAGES_FOLDER, f"{base_filename}.jpg")
        cv2.imwrite(image_path, g.current_image)
        
        # Charger le résultat de la segmentation
        seg_result = cv2.imread(result_path)
        if seg_result is None:
            return jsonify({"error": "Failed to load segmentation result"}), 500
            
        # Convertir en HSV pour extraire les masques par couleur
        hsv = cv2.cvtColor(seg_result, cv2.COLOR_BGR2HSV)
        
        # Masque pour les puces (vert)
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        puces_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Masque pour les bulles (rouge)
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        bulles_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        bulles_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        bulles_mask = cv2.bitwise_or(bulles_mask1, bulles_mask2)
        
        # Générer les annotations YOLO
        annotations = []
        
        # Puces (classe 0)
        puces_annotations = mask_to_yolo_polygon(puces_mask, 0)
        annotations.extend(puces_annotations)
        
        # Bulles (classe 1)
        bulles_annotations = mask_to_yolo_polygon(bulles_mask, 1)
        annotations.extend(bulles_annotations)
        
        # Sauvegarder les annotations
        label_path = os.path.join(TRAIN_LABELS_FOLDER, f"{base_filename}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        print(f"Saved training data: {image_path} and {label_path}")
        
        # Recharger et réentraîner le modèle YOLO
        if g.yolo_model is not None:
            print("Starting YOLO training...")
            g.yolo_model.train(
                data=os.path.join(DATA_FOLDER, 'config.yaml'),
                epochs=25,
                batch=16,
                device="cpu",
                exist_ok=True
            )
            print("YOLO training completed")
        else:
            print("Warning: YOLO model not available for training")
            
        return jsonify({
            "message": "Data saved and model retrained successfully",
            "image_path": image_path,
            "label_path": label_path,
            "annotations_count": len(annotations)
        })
        
    except Exception as e:
        print(f"Error in save_for_training: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



@app.route('/reset_segmentation', methods=['POST'])
def reset_segmentation():
    try:
        g.sam_masks = None
        g.current_mask_combined = None
        
        # Reset visualization
        if g.current_image is not None:
            result_path = os.path.join(app.config['SAM_FOLDER'], 'current_result.jpg')
            if os.path.exists(result_path):
                os.remove(result_path)  # Remove existing result
            cv2.imwrite(result_path, g.current_image)
            
            return jsonify({
                "message": "Segmentation reset successfully",
                "result_image": os.path.join('static', 'sam_output', 'current_result.jpg')
            })
        return jsonify({"message": "No image to reset"}), 400
    except Exception as e:
        print(f"Reset error: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route('/seg-page')
def segmentation_page():
    if not g.uploaded_image_path:
        return "No image uploaded", 400

    try:
        if g.sam is None or g.predictor is None:
            g.sam, g.predictor = load_sam_model()
            if g.sam is None or g.predictor is None:
                return "Failed to initialize SAM model", 500
        
        return render_template('segmentation_page.html',
                             image_path=g.uploaded_image_path,
                             result_path=os.path.join('static', 'sam_output', 'current_result.jpg'))
    except Exception as e:
        print(f"Error in segmentation page: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return f"Error preparing segmentation page: {str(e)}", 500
    


@app.route('/predict_yolo', methods=['POST'])
def predict_yolo():
    if g.yolo_model is None:
        g.yolo_model = load_yolo_model()
        if g.yolo_model is None:
            return jsonify({"error": "Failed to load YOLO model"}), 500
            
    if not g.uploaded_image_path or not os.path.exists(g.uploaded_image_path):
        return jsonify({"error": "No image uploaded"}), 400
        
    try:
        # Define fixed output path
        prediction_path = os.path.join(app.config['YOLO_FOLDER'], 'current_prediction.jpg')
        
        # Remove existing prediction if it exists
        if os.path.exists(prediction_path):
            os.remove(prediction_path)
        
        # Run prediction
        results = g.yolo_model.predict(g.uploaded_image_path, conf=0.25)
        result = results[0]
        
        # Save the prediction visualization
        result.save(prediction_path)
        
         # Process detections
        components_data = []  # To store the final table rows
        
        if result.masks is None:
            raise ValueError("The model did not produce any masks. Ensure the correct segmentation model is loaded.")
        
        
        # Generate masks and extract areas using OpenCV
        masks = result.masks.data.cpu().numpy()  # Get masks from YOLO results
        image_name = os.path.basename(g.uploaded_image_path)

        for idx, mask in enumerate(masks):  # Iterate through each mask
            # Calculate component area
            component_area = np.sum(mask)  # Count non-zero pixels in the mask
            
            # Calculate voids within the component
            voids = []  # Store areas of individual voids
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                void_area = cv2.contourArea(contour)
                voids.append(void_area)
            
            total_void_area = sum(voids)
            max_void_area = max(voids) if voids else 0
            void_percentage = total_void_area / component_area if component_area > 0 else 0
            max_void_percentage = max_void_area / component_area if component_area > 0 else 0

            # Add row to the table
            components_data.append({
                "Image": image_name,
                "Component": int(idx + 1),
                "Area": int(component_area),
                "Void %": round(float(void_percentage), 2),
                "Max. Void %": round(float(max_void_percentage), 2)
            })
        
        # Return the table as JSON
        return jsonify({
            "message": "YOLO analysis completed",
            "result_image": os.path.join('static', 'yolo_output', 'current_prediction.jpg'),
            "table": components_data
        })
        
    except Exception as e:
        print(f"YOLO prediction error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    


if __name__ == "__main__":
    try:
        print("Initializing models...")
        g.sam, g.predictor = load_sam_model()
        g.yolo_model = load_yolo_model()
        
        if g.sam is None or g.predictor is None:
            print("Warning: Failed to load SAM model at startup")
        if g.yolo_model is None:
            print("Warning: Failed to load YOLO model at startup")
            
        app.run(debug=True)
    except Exception as e:
        print(f"Startup error: {str(e)}")
        print("Traceback:", traceback.format_exc())



