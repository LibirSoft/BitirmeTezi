import cv2
import numpy as np
import glob
import random

print("Model Agırlıkları Yukleniyor")
model = cv2.dnn.readNetFromDarknet("yolov3.cfg","silahlar.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
print("Model Agırlıkları Yuklendi")
nesneler = ["silah"]
resim_yollari = glob.glob("./silahlarFoto/*.jpg")
print("Resimler Atandı")
random.shuffle(resim_yollari)

for resim_yol in resim_yollari:
    resim = cv2.imread(resim_yol)
    resim = cv2.resize(resim, (500, 500))
    height, width, channels = resim.shape
    blob = cv2.dnn.blobFromImage(resim, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_name = model.getUnconnectedOutLayersNames()
    layerOutputs = model.forward(output_layers_name)

    boxes = []
    guven_degerleri_list = []
    nesne_id_list = []
    for output in layerOutputs:
        for tespit in output:
            score = tespit[5:]
            class_id = np.argmax(score)
            guven_degeri = score[class_id]
            if guven_degeri > 0.5:
                center_x = int(tespit[0] * height)
                center_y = int(tespit[1] * width)
                w = int(tespit[2] * height)
                h = int(tespit[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                guven_degerleri_list.append((float(guven_degeri)))
                nesne_id_list.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, guven_degerleri_list, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            etiket = str(nesneler[nesne_id_list[i]])
            cv2.rectangle(resim, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resim, etiket, (x, y+30), font, 2, (0, 0, 255), 2)

    cv2.imshow("Silahlar", resim)
    cv2.waitKey()
