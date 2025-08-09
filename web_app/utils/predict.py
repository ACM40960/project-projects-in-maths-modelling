import onnxruntime as ort
import numpy as np
import cv2
import json

class WildlifePredictor:
    def __init__(self, md_model_path="../../models/megadetectorv6.onnx", 
                 cls_model_path="../../models/convnext_classifier.onnx"):
        """
        Initialize the wildlife predictor with ONNX models
        """
        # Load ONNX models
        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in available
                    else ["CPUExecutionProvider"])

        self.md_sess  = ort.InferenceSession(str(md_model_path),  providers=providers)
        self.cls_sess = ort.InferenceSession(str(cls_model_path), providers=providers)
        
        # Category mapping
        self.CATEGORY_NAME_TO_ID = {
            "bobcat": 6,
            "opossum": 1,
            "coyote": 9,
            "raccoon": 3,
            "bird": 11,
            "dog": 8,
            "cat": 16,
            "squirrel": 5,
            "rabbit": 10,
            "skunk": 7,
            "rodent": 99,
            "badger": 21,
            "deer": 34,
            "car": 33
        }
        
        self.CATEGORY_ID_TO_NAME = {v: k for k, v in self.CATEGORY_NAME_TO_ID.items()}
        
        # Classifier class list (excluding car)
        self.CLASSIFIER_CLASSES = [
            "badger", "bird", "bobcat", "cat", "coyote", "deer", "dog", "opossum",
            "rabbit", "raccoon", "rodent", "skunk", "squirrel"
        ]
        
        # Thresholds
        self.MD_CONF_THRESH = 0.35
        self.CLS_CONF_THRESH = 0.55
        
        # Get classifier input name
        self.cls_input_name = self.cls_sess.get_inputs()[0].name
        
        print("Wildlife predictor initialized successfully.")
    
    def clip_box(self, x1, y1, x2, y2, img_w, img_h):
        """Clip bounding box to image boundaries"""
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(0, min(int(x2), img_w - 1))
        y2 = max(0, min(int(y2), img_h - 1))
        return x1, y1, x2, y2
    
    def preprocess_crop(self, img):
        """Preprocess crop for classification"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Direct resize to 224x224
        resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        norm = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std
        
        # Convert to CHW format and add batch dimension
        transposed = np.transpose(norm, (2, 0, 1))
        return transposed[np.newaxis, :, :, :]
    
    def softmax(self, x):
        """Apply softmax to get probabilities"""
        e_x = np.exp(x - np.max(x))  # numerical stability
        return e_x / e_x.sum()
    
    def run_megadetector(self, img):
        """Run MegaDetector on image"""
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess for MegaDetector
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, :, :, :]
        
        # Run inference
        outputs = self.md_sess.run(None, {"images": img_input})
        raw = outputs[0][0]  # shape: (N, 6)
        
        # Scale factors to convert back to original size
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        detections = []
        for det in raw:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.MD_CONF_THRESH:
                continue
                
            cls_id = int(cls_id)
            if cls_id == 0:
                cls_name = "animal"
            elif cls_id == 1:
                cls_name = "vehicle"
            else:
                continue  # skip unknowns
            
            # Rescale to original size
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf),
                "class": cls_name,
                "class_id": cls_id
            })
        
        return detections
    
    def classify_animal(self, img, bbox):
        """Classify animal crop"""
        x1, y1, x2, y2 = bbox
        H, W = img.shape[:2]
        
        # Clip bbox to image boundaries
        x1, y1, x2, y2 = self.clip_box(x1, y1, x2, y2, W, H)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract crop
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # Preprocess crop
        inp = self.preprocess_crop(crop)
        
        # Run classification
        logits = self.cls_sess.run(None, {self.cls_input_name: inp})[0][0]
        probs = self.softmax(logits.astype(np.float32))
        
        cls_idx = np.argmax(probs)
        conf = probs[cls_idx]
        
        name = self.CLASSIFIER_CLASSES[cls_idx]
        coco_id = self.CATEGORY_NAME_TO_ID[name]
        
        return {
            "category": name,
            "category_id": coco_id,
            "conf": float(conf)
        }
    
    def predict(self, image_path):
        """
        Main prediction function
        Args:
            image_path: Path to image file
        Returns:
            List of predictions with bbox, category, category_id, conf
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run MegaDetector
        detections = self.run_megadetector(img)
        
        results = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            if det["class"] == "vehicle":
                # Vehicle detection - no classification needed
                pred = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "category": "car",
                    "category_id": 33,
                    "conf": det["conf"]
                }
                results.append(pred)
                
            elif det["class"] == "animal":
                # Animal detection - classify species
                cls_result = self.classify_animal(img, bbox)
                if cls_result is None:
                    continue
                
                # Only keep predictions above classification threshold
                if cls_result["conf"] >= self.CLS_CONF_THRESH:
                    pred = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "category": cls_result["category"],
                        "category_id": cls_result["category_id"],
                        "conf": cls_result["conf"]
                    }
                    results.append(pred)
        
        return results

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict wildlife in image')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--md_model', default='models/megadetectorv6.onnx', 
                       help='Path to MegaDetector ONNX model')
    parser.add_argument('--cls_model', default='models/convnext_classifier.onnx',
                       help='Path to classifier ONNX model')
    parser.add_argument('--output', '-o', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WildlifePredictor(args.md_model, args.cls_model)
    
    # Run prediction
    predictions = predictor.predict(args.image_path)
    
    print(f"Found {len(predictions)} predictions:")
    for i, pred in enumerate(predictions):
        bbox = pred['bbox']
        print(f"  {i+1}: {pred['category']} (conf: {pred['conf']:.3f}) "
              f"at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {args.output}")
    
    return predictions

if __name__ == "__main__":
    main()