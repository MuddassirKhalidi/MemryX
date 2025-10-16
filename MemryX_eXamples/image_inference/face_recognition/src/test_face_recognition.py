#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path to import MXFace
sys.path.append('.')

from MXFace import MXFace, AnnotatedFrame

def test_face_recognition():
    print("Testing MemryX Face Recognition with Simulator...")
    
    try:
        # Initialize MXFace with simulator mode
        print("Initializing MXFace...")
        mx_face = MXFace('../models')
        print("✓ MXFace initialized successfully!")
        
        # Create a simple test image (random RGB image)
        print("Creating test images...")
        test_image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_image2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"Test image 1 shape: {test_image1.shape}")
        print(f"Test image 2 shape: {test_image2.shape}")
        
        # Run inference
        print("Running face recognition inference...")
        
        # Process first image
        print("Processing image 1...")
        mx_face.put(test_image1)
        annotated_frame_1 = mx_face.get()
        print(f"✓ Image 1 processed. Detections: {annotated_frame_1.num_detections}")
        
        # Process second image
        print("Processing image 2...")
        mx_face.put(test_image2)
        annotated_frame_2 = mx_face.get()
        print(f"✓ Image 2 processed. Detections: {annotated_frame_2.num_detections}")
        
        # Calculate similarity if faces were detected
        if annotated_frame_1.num_detections > 0 and annotated_frame_2.num_detections > 0:
            face_embedding_1 = annotated_frame_1.detected_faces[0].embedding
            face_embedding_2 = annotated_frame_2.detected_faces[0].embedding
            
            similarity = MXFace.cosine_similarity(face_embedding_1, face_embedding_2)
            verified = similarity >= MXFace.cosine_threshold
            
            print(f"✓ Face similarity: {similarity:.4f}")
            print(f"✓ Verified as same person: {verified}")
        else:
            print("⚠ No faces detected in one or both images")
        
        # Cleanup
        mx_face.stop()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_recognition()
