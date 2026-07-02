import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Caricamento dataset

df = pd.read_csv('data.csv', sep=';')
df.columns = df.columns.str.strip()

# Rimozione classe Enrolled e feature del 2° semestre

df = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
second_sem_features = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]
df.drop(columns=second_sem_features, inplace=True, errors='ignore')

# Feature numeriche e categoriche

numeric_features = [
    'Age at enrollment',
    'Previous qualification (grade)',
    'Admission grade',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

categorical_features = [
    'Marital status', 'Nacionality', 'Displaced', 'Gender', 'International',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Scholarship holder'
]

X = df[numeric_features + categorical_features]
y = df['Target'].map({'Graduate': 0, 'Dropout': 1})  # Dropout = classe positiva

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing

tree_preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

nb_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Modelli

models = {
    'GaussianNB': Pipeline([
        ('preprocessor', nb_preprocessor),
        ('classifier', GaussianNB())
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', tree_preprocessor),
        ('classifier', DecisionTreeClassifier(criterion='gini', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', tree_preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
}

# Valutazione

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'\n=== {name} ===')
    print('Accuracy:          ', round(accuracy_score(y_test, y_pred), 4))
    print('Balanced Accuracy: ', round(balanced_accuracy_score(y_test, y_pred), 4))
    print('Precision Dropout: ', round(precision_score(y_test, y_pred, pos_label=1), 4))
    print('Recall Dropout:    ', round(recall_score(y_test, y_pred, pos_label=1), 4))
    print('F1-score Dropout:  ', round(f1_score(y_test, y_pred, pos_label=1), 4))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)
