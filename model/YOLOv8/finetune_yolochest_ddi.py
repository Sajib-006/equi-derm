from ultralyticsplus import YOLO, postprocess_classify_output

# load model
# model = YOLO('yolov8n-cls.pt') 
# model = YOLO('yolov8x-cls.pt') 

# # Train the model
# # model = YOLO('keremberke/yolov8m-chest-xray-classification')
# model = YOLO('yolov8n-cls.pt') 
# model.overrides['conf'] = 0.25  # model confidence threshold
# results = model.train(data='/home/sajib/RETFound_MAE/DDI_HAM10000', epochs=100, imgsz=224)


# # # perform inference
# # results = model.predict(data='/home/sajib/RETFound_MAE/DDI_data/test/benign')
# # observe results
# results = model('/home/sajib/RETFound_MAE/DDI_data/test/benign/000046.png') 
# print(results[0].probs) # [0.1, 0.2, 0.3, 0.4]
# processed_result = postprocess_classify_output(model, result=results[0])
# print(processed_result) # {"cat": 0.4, "dog": 0.6}


# prediction
model = YOLO('ChexNet/runs/classify/train9/weights/best.pt')
model.overrides['conf'] = 0.25  # model confidence threshold
import os
import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

benign_images = 'RETFound_MAE/DDI_data/test/benign'
malignant_images = 'RETFound_MAE/DDI_data/test/malignant'

# Iterate over each image in the directory
labels = []
pred_labels = []
for filename in os.listdir(benign_images):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image = os.path.join(benign_images, filename)
        pred = model(image) 
        processed_result = postprocess_classify_output(model, result=pred[0])
        # print(processed_result, type(processed_result)) # {"cat": 0.4, "dog": 0.6}
        if processed_result['benign'] > processed_result['malignant']:
            predicted_label = 0  # Benign is greater
        else:
            predicted_label = 1
        labels.append(0)
        pred_labels.append(predicted_label)

print(len(pred_labels), len(labels))
        
for filename in os.listdir(malignant_images):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image = os.path.join(malignant_images, filename)
        pred = model(image) 
        processed_result = postprocess_classify_output(model, result=pred[0])
        # print(processed_result, type(processed_result)) # {"cat": 0.4, "dog": 0.6}
        if processed_result['benign'] > processed_result['malignant']:
            predicted_label = 0  # Benign is greater
        else:
            predicted_label = 1
        labels.append(1)
        pred_labels.append(predicted_label)
print(len(pred_labels), len(labels))

# Calculate metrics
accuracy = accuracy_score(labels, pred_labels)
f1 = f1_score(labels, pred_labels)
cr = classification_report(labels, pred_labels)

print(f'Average Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Classification Report: {cr}')
