from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from wtforms.validators import DataRequired, Optional
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, SelectField, FloatField, IntegerField, SubmitField, FileField


model_types = [
    ('GB', 'Градиентный бустинг'),
    ('RF', 'Случайный лес')
]



class NewEnsembleForm(FlaskForm):
    name = StringField('Название модели', validators=[DataRequired()])
    model_type = SelectField('Тип ансамбля', choices=model_types)
    choose = SubmitField("Выбрать модель")

class HyperParamForm(FlaskForm):
    n_estimators = IntegerField('Количество деревьев', default=100)
    learning_rate = FloatField('Темп обучения', default=0.1)

    max_depth = IntegerField(
        'Максимальная глубина',
        default=5, 
        validators=[Optional()]
    )

    feature_subsample_size = IntegerField(
        'Размерность подвыборки признаков для одного дерева',
        default=None,
        validators=[Optional()],
    )

    random_state = IntegerField('Сид', default=42)

    trees_parameters = StringField(
        'Дополнительные параметры для дерева (JSON)',
        validators=[Optional()]
        #filters=[json_field_filter]
    )

    choose = SubmitField('Указать параметры')

class UploadForm(FlaskForm):
    name = StringField('Имя датасета', validators=[DataRequired()])
    features_file = FileField('Датасет (csv)', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV only!')
    ])
    target_name = StringField('Имя целевой переменной', validators=[
        Optional(),
    ])
    target_file = FileField('Целевая переменная (csv)', validators=[
        Optional(),
        FileAllowed(['csv'], 'CSV only!')
    ])
    choose = SubmitField('Выбрать датасет')