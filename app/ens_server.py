import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField
from forms import NewEnsembleForm, HyperParamForm, UploadForm

from ensembles import RandomForestMSE, GradientBoostingMSE

app = Flask(__name__, template_folder='templates')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'reallysecret'
Bootstrap(app)


models_list = []
model_name = []
pars = []
data = None
target = None
target_name = None

@app.route('/', methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():
    global model_name
    model = NewEnsembleForm()
    if model.validate_on_submit():
        models_list.append(model.data['model_type'])
        model_name.append(model.data['name'])
        return redirect('models')
    return render_template('index.html', form=model, models=models_list)


@app.route('/models/', methods=['GET', 'POST'])
def models():
    global pars, model_name
    form = HyperParamForm()
    if models_list[-1] == 'RF':
        del form.learning_rate
    if form.validate_on_submit():
        pars.append(form.data)
        return redirect(url_for('learning'))
    return render_template('models.html', form=form, name=model_name[-1], model_type=models_list[-1])


@app.route('/learning', methods=['GET', 'POST'])
def learning():
    global data, target, target_name
    form = UploadForm()
    if form.validate_on_submit():
        data = pd.read_csv(form.features_file.data, index_col=0,
                           float_precision='round_trip')
        target_name = form.target_name.data
        if form.target_file.data is not None:
            target = pd.read_csv(form.target_file.data, index_col=0,
                                 float_precision='round_trip')
            if target_name == '':
                target_name = target.columns[0]
        #model = NewEnsembleForm()
        return redirect(url_for('output'))#, form=model, models=models_list))
    return render_template('learning.html', form=form)


@app.route('/output', methods=['GET', 'POST'])
def output():
    global data, target
    regressor = []
    for i in range(len(models_list)):
        if models_list[i] == 'GB':
            regressor.append(GradientBoostingMSE(
                n_estimators=pars[i]['n_estimators'], 
                learning_rate=pars[i]['learning_rate'],
                max_depth=pars[i]['max_depth'],
                feature_subsample_size=pars[i]['feature_subsample_size']
                #trees_parameters=pars['trees_parameters']
            ))
        else:
            regressor.append(RandomForestMSE(
                n_estimators=pars[i]['n_estimators'], 
                max_depth=pars[i]['max_depth'],
                feature_subsample_size=pars[i]['feature_subsample_size']
                #trees_parameters=pars['trees_parameters']
            ))

   
    data = data.to_numpy()
    target = target.to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, target, test_size=0.15, random_state=pars[i]['random_state'])  
    
    for i in range(len(regressor)):
         
        reg, trace = regressor[i].fit(X_train, y_train, X_val, y_val)
        fig, ax = plt.subplots()
        sns.lineplot(x=np.arange(pars[i]['n_estimators']), y=trace['train_mses'], label='train', ax=ax)
        sns.lineplot(x=np.arange(pars[i]['n_estimators']), y=trace['test_mses'], label='validation', ax=ax)
        if models_list[i] == 'GB':
            ax.title.set_text(f"{models_list[i]} N={pars[i]['n_estimators']} lr={pars[i]['learning_rate']} depth={pars[i]['max_depth']}")
        else:
            ax.title.set_text(f"{models_list[i]} N={pars[i]['n_estimators']} depth={pars[i]['max_depth']}")
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('MSE')
        plt.savefig(f'app/static/new_plot{i}.png')

    return render_template('output.html', models=model_name)#, url='/images/new_plot.png')


 