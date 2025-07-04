# %%
import pickle
import datetime
import pandas as pd

from scipy.stats import randint, uniform, loguniform
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# %% 
df_processed = pd.read_csv('../data/processed/processed_cardio_data.csv')
target = 'flag_doenca_cardiaca'
cat_features = [col for col in df_processed.select_dtypes(include=['object']).columns if col != 'paciente_id'] + \
               [col for col in df_processed.select_dtypes(include=['int64', 'float64']).columns if col.startswith('flag_') and col not in ['paciente_id', target]]
num_features = [col for col in df_processed.select_dtypes(include=['int64', 'float64']).columns if not col.startswith('flag_') and col not in ['paciente_id', target]]

features = list(set(cat_features + num_features))
print("Variáveis categóricas: ", cat_features)
print("Variáveis numéricas: ", num_features)
print("Variável alvo: ", target)
# %%
seed = 22

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df_processed[features], 
    df_processed[target], 
    test_size=0.2, 
    random_state=42, 
    stratify=df_processed[target]
)

print("Taxa de resposta na base de treino: ", y_train.mean())
print("Taxa de resposta na base de teste: ", y_test.mean())

# %%
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first',sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
]).set_output(transform="pandas") 

# %%
models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=seed),
        {
            'classifier__C': loguniform(1e-4, 1e3),  
            'classifier__penalty': ['l2'],  
            'classifier__solver': ['liblinear', 'saga']  
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=seed),
        {
            'classifier__max_depth': randint(2, 100),  
            'classifier__min_samples_leaf': randint(1, 1000),  
            'classifier__criterion': ['gini', 'entropy', 'log_loss']  
        }
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=seed),
        {
            'classifier__n_estimators': randint(100, 2000),  
            'classifier__max_depth': randint(2, 100),
            'classifier__min_samples_leaf': randint(1, 1000),
            'classifier__criterion': ['gini', 'entropy', 'log_loss']
        }
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='logloss', random_state=seed),
        {
            'classifier__n_estimators': randint(100, 2000),
            'classifier__max_depth': randint(3, 30),
            'classifier__learning_rate': loguniform(1e-4, 0.5),
            'classifier__colsample_bytree': uniform(0.3, 0.7),  
            'classifier__reg_alpha': loguniform(1e-4, 10),  
            'classifier__reg_lambda': loguniform(1e-4, 10)
        }
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=seed),
        {
            'classifier__n_estimators': randint(50, 1000),
            'classifier__learning_rate': loguniform(1e-4, 1.0)
        }
    ),
    'LightGBM': (
        LGBMClassifier(random_state=seed),
        {
            'classifier__n_estimators': randint(100, 2000),
            'classifier__max_depth': randint(3, 50),
            'classifier__learning_rate': loguniform(1e-4, 0.5),
            'classifier__num_leaves': randint(20, 512),
            'classifier__colsample_bytree': uniform(0.3, 0.7),
            'classifier__reg_alpha': loguniform(1e-4, 10),
            'classifier__reg_lambda': loguniform(1e-4, 10)
        }
    )
}
# %%
def report_metrics(y_true, y_proba, cohort=0.5):

    y_pred = (y_proba[:, 1] > cohort).astype(int)

    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_proba[:, 1])
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)

    error_metrics = {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return error_metrics
# %%

def model_training(model_name, model, param_grid):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    search = model_selection.RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=seed
    )
    search.fit(X_train, y_train)

    y_pred_train = search.predict_proba(X_train)
    y_pred_test = search.predict_proba(X_test)

    train_result = report_metrics(y_train, y_pred_train)
    test_result = report_metrics(y_test, y_pred_test)
    
    result = {
        'model': search.best_estimator_,
        "features": features,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'train_metrics': train_result,
        'test_metrics': test_result,
        "dt_training": datetime.datetime.now()
    }

    return result

# %%
results = {}

for model_name, (model, param_grid) in models.items():
    print(f'Treinando: {model_name}')
    result = model_training(model_name, model, param_grid)
    results[model_name] = result
# %%
results
# %%
file_path = '../data/predicted/model_series.pkl'

with open(file_path, 'wb') as f:
    pickle.dump(results, f)