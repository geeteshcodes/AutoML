import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA

#CLASSIFIERS
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

#
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

def build_preprocessor(X):

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    ordinal_features = []
    onehot_features = []

    for col in categorical_features:
        if X[col].nunique() <= 5:
            onehot_features.append(col)
        else:
            ordinal_features.append(col)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler()),
    ])

    onehot_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('cat', onehot_transformer, onehot_features)
    ])

    return preprocessor

def train_automl(df, target, task="classification"):

    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression())
    ])

    if task == "classification":
        param_grid = [
            {'model':[LogisticRegression(max_iter=1000)],'model__C':[0.1,1,10]},
            {'model':[RandomForestClassifier()],'model__n_estimators':[50,100]},
            {'model':[SVC()],'model__C':[0.1,1],'model__kernel':['linear','rbf']},
            {'model':[KNeighborsClassifier()],'model__n_neighbors':[3,5]},
            {'model':[GradientBoostingClassifier()]},
            {'model':[AdaBoostClassifier()]},
            {'model':[ExtraTreesClassifier()]},
            {'model':[SGDClassifier()]},
            {'model': [XGBClassifier(eval_metric='logloss')]}
        ]
        scoring = "accuracy"

    else:
        param_grid = [
            {'model':[LinearRegression()]},
            {'model':[Ridge()], 'model__alpha':[0.1,1,10]},
            {'model':[Lasso()], 'model__alpha':[0.1,1]},
            {'model':[RandomForestRegressor()], 'model__n_estimators':[50,100]},
            {'model':[GradientBoostingRegressor()]},
            {'model':[ExtraTreesRegressor()]},
            {'model':[KNeighborsRegressor()]},
            {'model':[XGBRegressor()]}
        ]
        scoring = "r2"

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    model_name = type(best_model.named_steps["model"]).__name__

    return best_model, grid.best_score_, model_name

