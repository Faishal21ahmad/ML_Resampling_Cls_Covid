import json
import numpy as np
import pandas as pd
import csv
import pickle
from numpy import False_
import matplotlib.pyplot as plt
import os

from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score
from imblearn.metrics import classification_report_imbalanced
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["SECRET_KEY"] = "AcrMlCvd19"

# ================ Path Apps ======================== #
pth = '/home/fais/Documents/Repo/ML_Resampling_Cls_Covid'


path_input = pth + '/data_input/'
path_label = pth + '/label_list/'
path_model = pth + '/model_training/'
path_result = pth + '/result/'

# ================ extensi permis ================= #
ALLOWED_EXTENSIONS = set(['csv'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ================ Cari data ================= #
def caridata(array, target):
    for i, elemen in enumerate(array):
        if elemen == target:
            return 1
    return 0


# ================ Cari .pkl data ================= #
def scan_pkl_files(folder_path):
    file_info_list = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith(".pkl"):
            file_info = {
                "filename": filename
            }
            file_info_list.append(file_info)
    return file_info_list


# ================ Cari .csv data ================= #
def scan_csv_files(folder_path):
    file_info_list = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith(".csv"):
            file_info = {
                "filename": filename
            }
            file_info_list.append(file_info)
    return file_info_list

# ================ ROOT Accurasy ================= #

@app.route('/', methods=['GET', 'POST'])
def root():
    title = "Accuracy"
    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            session.pop('filename', None)
            file = request.files['file']

            if 'file' not in request.files or not request.files['file']:
                return render_template('accuracy.html', title=title, fil=0)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                new_filename = f'{filename.split(" . ")[0]}'
                file.save(os.path.join('data_input', new_filename))
            session["filename"] = new_filename
            pat = "data_input/" + new_filename
            df = pd.read_csv(pat)

            co = []
            for item in df:
                co.append(item)

            data = []
            with open(pat, "r") as f:
                content = f.readlines()
                for line in content[0:]:
                    columns = line.strip().split(",")
                    h = np.count_nonzero(columns)
                    if h == False:
                        i = 0
                    else:
                        i = h
                    data.append(columns)

            session["colLab"] = co

            # retrun if post number 1
            return render_template('accuracy.html', title=title, fil=new_filename, data=data, co=co, i=i)
        
        
        
        if proses == "2":
            cls = request.form.get('kls')

            if 'kls' not in request.form or not request.form.get('kls'):
                title = "Accuracy Data Input"
                return render_template('accuracy.html', title=title, cls=0, fil=0)

            co = session["colLab"]
            fil = session["filename"]

            df = pd.read_csv("data_input/" + fil)

            fit = [value for value in co if value not in cls]

            fil = fil[:-4] if fil.endswith('.csv') else fil

            pickle.dump(fit, open(f"" + path_label + fil + ".pkl", "wb"))

            data = []
            with open("data_input/" + fil + ".csv", "r") as f:
                content = f.readlines()
                for line in content[0:]:
                    columns = line.strip().split(",")
                    i = np.count_nonzero(columns)
                    data.append(columns)

            X = df.loc[:, fit].values
            y = df.loc[:, cls].values

            smote = SMOTE(random_state=50)
            adasyn = ADASYN(random_state=50)
            X_res_smote, y_res_smote = smote.fit_resample(X, y)
            X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X, y)

            reps = [
                ('Oriset', y, X),
                ('Smote', y_res_smote, X_res_smote),
                ('Adasyn', y_res_adasyn, X_res_adasyn)
            ]

            resamcount = []
            for rep in reps:
                count = Counter(rep[1])
                resamcount.append([rep[0], count])

            tss = [
                ('TS_90%', 0.1),
                ('TS_80%', 0.2),
                ('TS_70%', 0.3)
            ]

            test_trains = []
            for ts in tss:
                for rep in reps:
                    resultTT = X_train, X_test, y_train, y_test = train_test_split(
                        rep[2], rep[1], test_size=ts[1], random_state=50)
                    test_trains.append([rep[0], ts[0], resultTT])

            models = [
                ('SVM', SVC()),
                ('RF', RandomForestClassifier()),
                ('NN', MLPClassifier())
            ]

            model_trains = []
            for model in models:
                for test_train in test_trains:
                    result_model_train = model[1].fit(
                        test_train[2][0], test_train[2][2])
                    model_trains.append([model[0], test_train[0], test_train[1],
                                        test_train[2][1], test_train[2][3], result_model_train])

            model_tok = []
            for model in model_trains:
                model_tok.append([model[0]+"_"+model[1]+"_"+model[2], model[5]])
            pickle.dump(model_tok, open(
                f"" + path_model + fil + ".pkl", "wb"))

            preds = []
            for model_train in model_trains:
                result_predi = model_train[5].predict(model_train[3])
                preds.append([model_train[0], model_train[1],
                            model_train[2], model_train[4], result_predi])

            result = []
            for pred in preds:
                acr = round(accuracy_score(pred[3], pred[4])*100, 2)
                report = classification_report_imbalanced(
                    pred[3], pred[4], output_dict=True)
                pre = round(report['avg_pre']*100, 2)
                rec = round(report['avg_rec']*100, 2)
                spe = round(report['avg_spe']*100, 2)
                f1 = round(report['avg_f1']*100, 2)
                geo = round(report['avg_geo']*100, 2)
                iba = round(report['avg_iba']*100, 2)
                sup = round(report['total_support']*100, 2)
                result.append([pred[0], pred[1], pred[2], acr,
                            pre, rec, spe, f1, geo, iba, sup])

            grub = []
            grub.append([fil, resamcount, result])
            pickle.dump(grub, open(
                f"" + path_result + fil +".pkl", "wb"))

            # retrun if post number 2
            return render_template('accuracy.html', title=title, data=data, i=i, co=co, resamcount=resamcount, result=result, fil=fil, cls=cls, fit=fit)

        # retrun if post not proses
        return render_template('accuracy.html', title=title)
    session.pop('filename', None)
    # retrun if get
    return render_template('accuracy.html', title=title)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    title = "Compare"
    file = scan_pkl_files(path_result)

    fis = []
    for fil in file:
        fis.append(fil["filename"])

    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            str_file = [str(x)for x in request.form.values()]
            str_file = np.delete(str_file, -1)
            str_file = str_file.tolist()
            session["str_file"] = str_file

            grub = []
            for name in str_file:
                grub.append(pickle.load(open('result/' + name, 'rb')))

            alg = []
            rs = []
            ts = []

            if alg == []:
                alg = ['SVM', 'RF', 'NN']

            if rs == []:
                rs = ['Oriset', 'Smote', 'Adasyn']

            if ts == []:
                ts = ['TS_90%', 'TS_80%', 'TS_70%']

            # return render_template('compare.html', title=title, file=fis, filed=str_file, grub="none")
            return render_template('compare.html', title=title, file=fis, filed=str_file, grub=grub, rs=rs, alg=alg, ts=ts)
        if proses == "2":
            file = scan_pkl_files(path_result)

            if "str_file" not in session:
                session["str_file"] = []

            str_file = session["str_file"]

            files = []
            for fil in file:
                files.append(fil["filename"])

            grub = []
            for name in str_file:
                grub.append(pickle.load(open('result/' + name, 'rb')))

            alg = [str(x) for x in request.form.getlist('algo')]
            rs = [str(x) for x in request.form.getlist('resam')]
            ts = [str(x) for x in request.form.getlist('dts')]

            if alg == []:
                alg = ['SVM', 'RF', 'NN']

            if rs == []:
                rs = ['Oriset', 'Smote', 'Adasyn']

            if ts == []:
                ts = ['TS_90%', 'TS_80%', 'TS_70%']
            return render_template('compare.html', title=title, filed=str_file, file=files, grub=grub, rs=rs, alg=alg, ts=ts)

    return render_template('compare.html', title=title, file=fis, grub="none")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    title = "Predict"
    file_info_list = scan_pkl_files(path_model)

    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            ti = request.form['sel']
            session["slt"] = ti
            label = pickle.load(open('label_list/' + ti, 'rb'))

            return render_template('predict.html', title=title, file=file_info_list, label=label, ti=ti)
            
        if proses == "2":
            ty = session["slt"]

            label = pickle.load(open('label_list/' + ty, 'rb'))
            models = pickle.load(open('model_training/' + ty, 'rb'))

            rs = []
            float_feature = [int(x) for x in request.form.values()]
            feature = np.delete(float_feature, -1)
            feature = feature.tolist()
            feature = [np.array(feature)]

            for itm in models:
                pr = itm[1].predict(feature)
                if pr == 1:
                    f = "= Positif Covid"
                else:
                    f = "= Tidak Covid"
                rs.append([itm[0], f])

            return render_template('predict.html', title=title, file=file_info_list, label=label, ti=ty, rs=rs, feature=feature, pr=models)


    return render_template('predict.html', title=title, file=file_info_list)

@app.route('/file', methods=['GET', 'POST'])
def file():
    title = "File"
    
    file = scan_csv_files(path_input)

    if request.method == 'POST':
        itd = request.form['itd']
        itd = itd[:-4] if itd.endswith('.csv') else itd

        file_path = path_input + itd + ".csv"
        label = path_label + itd + ".pkl"
        model = path_model + itd + ".pkl"
        result = path_result + itd + ".pkl"

        if os.path.exists(file_path):
            os.remove(file_path)
            os.remove(label)
            os.remove(model)
            os.remove(result)
            file = scan_csv_files(path_input)
            return render_template('file.html', title=title, file=file, alert=(f"Data" + itd + " Berhasil Di hapus"))
        else:
            return render_template('file.html', title=title, file=file, alert=(f"Data" + itd + " Tidak Ditemukan"))

    return render_template('file.html', title=title, file=file)


if __name__ == '__main__':
    app.run(port=3065, debug=True)
