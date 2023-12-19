import cv2
import imutils
import numpy as np
from flask import Flask, render_template, request
import base64

app = Flask(__name__)

def process_image(file_storage):
    img = cv2.imdecode(np.frombuffer(file_storage.read(), np.uint8), -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfiller = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfiller, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    _, buffer = cv2.imencode('.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_str = process_image(file)
            return render_template('index.html', img_str=img_str)

    return render_template('index.html', img_str=None)

if __name__ == '__main__':
    app.run(debug=True)
