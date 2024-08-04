from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = capture.read()
        if not success:
            break
        else:
            faces = face_cascade.detectMultiScale(img, 1.2, 4)
            for (x, y, w, h) in faces:
                face_region = img[y:y + h, x:x + w]
                gaussian_blur = cv2.GaussianBlur(face_region, (91, 91), 0)
                img[y:y + h, x:x + w] = gaussian_blur

            if len(faces) == 0:
                cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global capture
    capture = cv2.VideoCapture(0)
    return "Started"

@app.route('/stop')
def stop():
    global capture
    capture.release()
    return "Stopped"

if __name__ == '__main__':
    app.run(debug=True)
