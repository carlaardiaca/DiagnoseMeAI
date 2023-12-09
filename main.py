from data_preprocessing import *
from clean_data import clean_text
from sklearn.model_selection import train_test_split
from model_selection import train_random_forest, save_best_model
import pandas as pd
from model_testing import print_classification_report, plot_confusion_matrix, lime_explain_misclassified, make_predict_probaLIME, evaluate_top3_accuracy_and_explain
from sklearn.metrics import confusion_matrix
import seaborn as sns
from lime.lime_text import LimeTextExplainer
from joblib import dump


def split_data(df):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('label', axis=1),
        df['label'],
        test_size=0.20,
        random_state=111,
        stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

# Load data
df = pd.read_csv('data/Symptom2Disease.csv')

# Data Cleaning
df['text'] = df['text'].apply(clean_text)
print("OK_data")

# Clean and preprocess text data
df['text'] = df['text'].apply(remove_irrelevant_words)

# Data Splitting
X_train, X_test, y_train, y_test = split_data(df)
LIME = X_test['text']
# Preprocessing
# Create tokenizer based on the training data
tokenizer = create_tokenizer(X_train)
dump(tokenizer, 'paraules_freq.joblib')

# Tokenize the text in the training and testing data
X_train['text'] = X_train['text'].apply(lambda x: apply_tokens(x, tokenizer))

# Determine the maximum length for padding
max_length = max(len(seq) for seq in X_train['text'])

# Apply padding to the tokenized training and testing data
X_train['text'] = apply_padding(X_train['text'], max_length)

# Apply one-hot encoding to the padded training and testing data
num_words = len(tokenizer)
X_train_encoded = one_hot_encode(X_train['text'], num_words)

# Encode the labels
y_train_encoded, label_names = encode_labels(y_train)
print("OK_preprocessing")

# Train the best model (Random Forest in this case)
best_rf_model = train_random_forest(X_train_encoded, y_train_encoded)
print("OK_train")

# Save the best model
save_best_model(best_rf_model, 'best_random_forest_model.joblib')
print("OK trainning model")

# Prepare test data
# Tokenization
print(X_test['text'])

X_test['text'] = X_test['text'].apply(lambda x: apply_tokens(x, tokenizer))
print(X_test['text'])
# Padding
X_test['text'] = apply_padding(X_test['text'], max_length)

# One-Hot Encoding
X_test_encoded = one_hot_encode(X_test['text'], len(tokenizer))

# Convert encoded data to DataFrame
df_matriu_test = pd.DataFrame(X_test_encoded)
paraules = sorted(tokenizer, key=tokenizer.get)
df_matriu_test.columns = paraules
df_matriu_test["INDEX"] = X_test.index
df_matriu_test = df_matriu_test.set_index('INDEX')
dump(paraules, 'paraules_dicc.joblib')

# Join with X_test and cleanup
X_test = X_test.join(df_matriu_test)
X_test = X_test.drop(['Unnamed: 0', 'text'], axis=1)  

# Prepare y_test
y_test = y_test.map(label_names)
dump(label_names, 'label_noms.joblib')

# Print classification report
y_pred = print_classification_report(best_rf_model, X_test, y_test)
print("OK prediction")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_matrix, df['label'].unique(), save=True)

# LIME explanations for misclassified cases
mal_classificades = []
for i, (y_test_i, y_pred_i) in enumerate(zip(y_test, y_pred)):
    if y_test_i != y_pred_i:
        mal_classificades.append(i)
        
label_names_list = label_names.keys()
nombre_a_numero = {v: k for k, v in label_names.items()}
explainer = LimeTextExplainer(class_names=label_names_list, verbose=True)
predict_probaLIME = make_predict_probaLIME(best_rf_model, tokenizer)
lime_explain_misclassified(explainer, mal_classificades, y_test, y_pred, predict_probaLIME, LIME, nombre_a_numero, 3)

top3_accuracy = evaluate_top3_accuracy_and_explain(explainer, best_rf_model, X_test, y_test, y_pred,predict_probaLIME, LIME, nombre_a_numero)