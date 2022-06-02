import cv2
import numpy as np

model = cv2.dnn.readNetFromDarknet("yolov3.cfg",
                                   "silahlar.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
nesneler = ["silah"]

video = cv2.VideoCapture("testVideo.mp4")
W = None
H = None

while 1:

    h_an, an = video.read()
    an = cv2.resize(an, (500, 500))
    an_h = an.shape[0]
    an_w = an.shape[1]
    rects = []
    blob = cv2.dnn.blobFromImage(an, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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
                center_x = int(tespit[0] * an_h)
                center_y = int(tespit[1] * an_w)
                w = int(tespit[2] * an_w)
                h = int(tespit[3] * an_h)
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
            cv2.rectangle(an, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(an, etiket, (x, y + 30), font, 2, (0, 0, 255), 2)

    cv2.imshow("Silahlar", an)
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
