import sys
import cv2
import numpy as np

model = cv2.dnn.readNetFromDarknet("/home/libir/Desktop/dice/bitirme/GunDetectService/src/YoloGunDetect/yolov3.cfg",
                                   "/home/libir/Desktop/dice/bitirme/GunDetectService/src/YoloGunDetect/silahlar.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
nesneler = ["silah"]
image_path = sys.argv[1]

resim = cv2.imread("/home/libir/Desktop/dice/bitirme/GunDetectService/images/" + image_path)
resim = cv2.resize(resim, (500, 500))
height, width, channels = resim.shape
blob = cv2.dnn.blobFromImage(resim, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)
output_layers_name = model.getUnconnectedOutLayersNames()
layerOutputs = model.forward(output_layers_name)

for output in layerOutputs:
    for tespit in output:
        score = tespit[5:]
        class_id = np.argmax(score)
        guven_degeri = score[class_id]
        if guven_degeri > 0.5:
            print("Gun detected.")
            break
    print("Gun not detected")
