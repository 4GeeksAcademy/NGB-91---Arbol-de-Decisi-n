from utils import db_connect
engine = db_connect()

# your code here
# Explore here
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import *
from imblearn.metrics import specificity_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# VER BASE DE DATOS:

df_diabetes = pd.read_csv('../data/raw/diabetes.csv')
df_diabetes
df_diabetes.shape
df_diabetes['Outcome'].value_counts()
df_diabetes.info()
df_diabetes.describe()
print("Al observar que los datos mínimos de prácticamente todos los valores es 0. Vamos a eliminar todos los datos cuyo valor sea 0)

# ELIMINAR VALORES 0:

df_diabetes.drop(df_diabetes[(df_diabetes['Pregnancies'] == 0) |
                             (df_diabetes['Glucose'] == 0) |
                             (df_diabetes['BloodPressure'] == 0) |
                             (df_diabetes['SkinThickness'] == 0) |
                             (df_diabetes['Insulin'] == 0) |
                             (df_diabetes['BMI'] == 0)].index, inplace=True)

df_diabetes


df_diabetes.describe()
df_diabetes.isnull().sum()

# VISUALIZACIÓN PARALELA DE LOS DATOS ORIGINALES:

plt.figure(figsize=(8, 5))
pd.plotting.parallel_coordinates(df_diabetes, 'Outcome', color=['indigo', 'red'])
plt.xticks(rotation=45)
plt.show()

# DATOS ESCALADOS:

df_diabetes = df_diabetes.reset_index(drop=True)

data_sc = pd.DataFrame(StandardScaler().fit_transform(df_diabetes.drop('Outcome', axis=1)), columns=df_diabetes.columns[:-1])
data_sc['Outcome'] = df_diabetes['Outcome']
data_sc

# VISUALIZACIÓN DE DATOS ESCALADOS:

plt.figure(figsize=(8, 5))
pd.plotting.parallel_coordinates(data_sc, 'Outcome', color=['indigo', 'red'])
plt.xticks(rotation=45)
plt.show()

# DIVISIÓN DE LOS DATOS:

X = df_diabetes.drop('Outcome', axis=1)
y = df_diabetes['Outcome']

print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ENTRENAMIENTO DE MODELOS:

# ÁRBOL SIMPLE:

simple_tree = DecisionTreeClassifier(max_depth=3, max_features=X_train.shape[1]//2, min_samples_leaf=20, min_samples_split=30, random_state=42)
simple_tree.fit(X_train, y_train)
# ÁRBOL COMPLEJO:

complex_tree = DecisionTreeClassifier(max_depth=100, min_samples_leaf=1, random_state=42)
complex_tree.fit(X_train, y_train)

# VISUALIZACIÓN DE LOS ÁRBOLES:

# VISUALIZACIÓN ÁRBOL SIMPLE:

plt.figure(figsize=(10, 6))
plot_tree(simple_tree, feature_names=X_train.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()
X_train[(X_train['Glucose'] > 16.795) & (X_train['BMI'] <= 0.073)]

# REPRESENTACIÓN TEXTUAL DEL ÁRBOL:

text_representation = export_text(simple_tree, feature_names=list(X_train.columns))
print(text_representation)

# FUNCIONES MÉTRICAS:

def get_metrics(y_train, y_test, y_pred_train, y_pred_test):
    # Calcular métricas para el conjunto de entrenamiento
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, y_pred_train)
    train_specificity = specificity_score(y_train, y_pred_train)

    # Calcular métricas para el conjunto de prueba
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_specificity = specificity_score(y_test, y_pred_test)

    # Calcular la diferencia entre métricas de entrenamiento y prueba
    diff_accuracy = train_accuracy - test_accuracy
    diff_f1 = train_f1 - test_f1
    diff_auc = train_auc - test_auc
    diff_precision = train_precision - test_precision
    diff_recall = train_recall - test_recall
    diff_specificity = train_specificity - test_specificity

    # Crear un DataFrame con los resultados
    metrics_df = pd.DataFrame([[train_accuracy, train_f1, train_auc, train_precision, train_recall, train_specificity],[test_accuracy, test_f1, test_auc, test_precision, test_recall, test_specificity],[diff_accuracy, diff_f1, diff_auc, diff_precision, diff_recall, diff_specificity]],
                              columns = ['Accuracy', 'F1', 'AUC', 'Precision', 'Recall', 'Specificity'],
                              index = ['Train','Test', 'Diferencia'])

    return metrics_df

# EVALUACIÓN DE MODELOS:

# Evaluar el modelo complejo en entrenamiento y prueba
train_pred_complex = complex_tree.predict(X_train)
test_pred_complex = complex_tree.predict(X_test)

# Evaluar el modelo simple en entrenamiento y prueba
train_pred_simple = simple_tree.predict(X_train)
test_pred_simple = simple_tree.predict(X_test)

# MATRIZ DE CONFUSIÓN:

cm = confusion_matrix(y_test,test_pred_complex)
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()
print("El modelo acertó 37 veces cuando dijo que no hay diabetes, y acertó 14 veces cuando dijo que sí hay diabetes, también se equivocó 9 veces diciendo que había diabetes cuando no la había, y 8 veces diciendo que no había cuando sí había")

# MÉTRICAS FINALES:

# Métricas del modelo complejo
get_metrics(y_train, y_test, train_pred_complex, test_pred_complex)
print("Árbol Complejo:")
print(get_metrics(y_train, y_test, train_pred_complex, test_pred_complex))

# Métricas del modelo simple
get_metrics(y_train, y_test, train_pred_simple, test_pred_simple)
print("Árbol Simple:")
print(get_metrics(y_train, y_test, train_pred_simple, test_pred_simple))

print(ÁRBOL COMPLEJO:)
print("En entrenamiento acierta todo, pero en test baja mucho: parece que se ha aprendido los datos de memoria.Tiene mucha diferencia entre train y test, lo que indica sobreajuste (no generaliza bien)")

print(ÁRBOL SIMPLE:)
print("Rinde parecido en entrenamiento y test, sin grandes diferencias. Aunque la precisión total es un poco menor, es más estable y generaliza mejor")

print(CONCLUSIÓN:)
print("Aunque el árbol complejo tiene mejores resultados en entrenamiento, no generaliza bien a nuevos datos. En cambio, el árbol simple ofrece un rendimiento más equilibrado y estable, por lo que es la mejor opción para predecir correctamente casos nuevos sin hacer trampas con los datos")