# app.py
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import urllib.request
import os, shutil
from werkzeug.utils import secure_filename
#from object_detection import *

import glob

from objectDetectionYolov3 import *

from flask_bootstrap import Bootstrap

app = Flask(__name__)

Bootstrap(app)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

folderUploads = 'static/uploads\\'
folderOutput = 'static/output'

#Funzione per eliminare i file dalla cartella
def deleteFileFolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


#funzione per prendere l'ultimo file da una cartella
def last_created_file(folder_path):
    list_of_files = glob.glob(folder_path + '/*')  # Ottieni tutti i file nella cartella specificata
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)  # Trova il file più recente per tempo di creazione
        return latest_file.replace(str(folder_path) + "\\", "")
    else:
        print("Nessun file trovato nella cartella:", folder_path)
        return None

deleteFileFolder(folderUploads)
deleteFileFolder(folderOutput)

# Immagine iniziale
immagine_iniziale = "../face_0_1.0.jpg"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html', immagine=immagine_iniziale)


#caricamento immagine
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        #prendo l'ultimo file dalla cartella uploads
        latest_file = last_created_file('static/uploads')

        filename = secure_filename(file.filename)

        #controllo se il nuovo file caricato è diverso dal vecchio,
        #se sono diversi, elimino l'output ottenuto in precedenza perchè la logica della pagina si basa sull'esistenza di tale file
        if(filename != latest_file):
            deleteFileFolder(folderOutput)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

#funzione per richiamare lo script detection
@app.route("/detect/")
def request_model_switch():
    detection()
    return "nothing"

#funzione per ottenere da script detection e visualizzare sulla pagina il numero di oggetti riconosciuti
@app.route('/oggettiRiconosciuti')
def stringa_da_python():
    stringa = os.environ["OGGETTI_RICONOSCIUTI"]
    return jsonify({'stringa': stringa})



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')