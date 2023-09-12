
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
predicted_folder = os.path.join('static', 'predicted')
 
app.config['UPLOAD'] = upload_folder
app.config['PREDICT'] = predicted_folder







model = YOLO('best.pt')

 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        # img = os.path.join(app.config['UPLOAD'], filename)

        name = os.path.join(app.config['UPLOAD'], filename)

        frame = cv2.imread(name)

        results = model(name)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        cv2.imwrite(os.path.join(app.config['PREDICT'], filename), frame)

        show_frame = os.path.join(app.config['PREDICT'], filename)

        return render_template('image_render.html', img=show_frame)
    
    return render_template('image_render.html')
 


@app.route('/api', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        # img = os.path.join(app.config['UPLOAD'], filename)

        name = os.path.join(app.config['UPLOAD'], filename)

        frame = cv2.imread(name)

        results = model(name)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        cv2.imwrite(os.path.join(app.config['PREDICT'], filename), frame)

        show_frame = os.path.join(app.config['PREDICT'], filename)

        return render_template('image_render.html', img=show_frame)
    
 
if __name__ == '__main__':
    app.run(debug=True, port=8001)