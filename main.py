from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from matplotlib import pyplot as plt
import imutils
import easyocr
import os
import numpy as np
import cv2
import os.path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'C:/Users/Akanksh shetty/Desktop/Black Box/PHOTOS'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("UPLOAD")


def detect_ANPR():
    if os.path.exists("C:/Users/Akanksh shetty/Desktop/Black Box/PHOTOS/sample_1.jpeg"):
        os.remove("C:/Users/Akanksh shetty/Desktop/Black Box/PHOTOS/sample_1.jpeg")
        return redirect(url_for('detect'))

    else:
        sample_data()
        img = cv2.imread('C:/Users/Akanksh shetty/Desktop/Black Box/PHOTOS/sample_1.jpeg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # noise reduction
        edged = cv2.Canny(bfilter, 100, 200)  # Edge detection
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

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
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)
        if len(result) > 1:
            text = result[0][-2]
            text1 = result[1][-2]
            return (text + text1)
        else:
            text = result[0][-2]
            return (text)


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data  # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        return redirect(url_for('detect'))
    return render_template('index.html', form=form)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    number = detect_ANPR()

    processed_data()

    return (number)

def sample_data(): # converts the img_file to sample_1
    folder = r"C:\Users\Akanksh shetty\Desktop\Black Box\PHOTOS/"
    count = 1
    # count increase by 1 in each iteration
    # iterate all files from a directory
    for file_name in os.listdir(folder):
        # Construct old file name
        source = folder + file_name

        # Adding the count to the new file name and extension
        destination = folder + "sample_" + str(count) + ".jpeg"

        # Renaming the file
        os.rename(source, destination)
        count += 1

def processed_data():  # converts the sales_1 to process_1
    folder = r"C:\Users\Akanksh shetty\Desktop\Black Box\PHOTOS/"
    count = 1
    # count increase by 1 in each iteration
    # iterate all files from a directory
    for file_name in os.listdir(folder):
        # Construct old file name
        source = folder + file_name

        # Adding the count to the new file name and extension
        destination = folder + "process_" + str(count) + ".jpeg"

        # Renaming the file
        os.rename(source, destination)
        count += 1


# images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]


if __name__ == '__main__':
    app.run(debug=True)
