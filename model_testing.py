import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from lime import lime_text
from data_preprocessing import apply_tokens
from datetime import datetime

def print_classification_report(grid_search, X_test, y_test):
    """
    Print the classification report for the test data.

    Args:
    grid_search: The trained model.
    X_test: Test data features.
    y_test: True labels for the test data.
    """
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_confusion_matrix(conf_matrix, labels, save=False):
    """
    Plot a confusion matrix.

    Args:
    conf_matrix: Confusion matrix data.
    labels: Labels for the confusion matrix axes.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'confusion_matrix_{timestamp}.png'
        plt.savefig(filename)
        print(f"Confusion Matrix saved as {filename}")
    plt.show()

def padding(token_indices, max_length):
    if len(token_indices) < max_length:
        token_indices += [0] * (max_length - len(token_indices))
    else:
        token_indices = token_indices[:max_length]
    return token_indices

def make_predict_probaLIME(grid_search, tokenizer):
    def predict_probaLIME(text_samples):
        all_predictions = []
        for texts in text_samples:
            words = texts.split()
            token_indices = apply_tokens(words, tokenizer)
            padded_indices = padding(token_indices, 27)

            one_hot_enc = np.zeros(len(tokenizer))
            for idx in padded_indices:
                if idx > 0 and idx < len(tokenizer):
                    one_hot_enc[idx - 1] = 1

            one_hot_vector_reshaped = one_hot_enc.reshape(1, -1)

            prediction = grid_search.predict_proba(one_hot_vector_reshaped)
            all_predictions.append(prediction[0])

        return np.array(all_predictions)
    return predict_probaLIME

def lime_explain_misclassified(explainer, mal_classificades, y_test, y_pred, predict_probaLIME, LIME, label_noms, num):
    """
    Explain misclassified cases using LIME.

    Args:
    explainer: LIME text explainer.
    mal_classificades: Indices of misclassified cases.
    y_test: True labels for the test data.
    y_pred: Predicted labels.
    predict_probaLIME: Function for prediction probabilities for LIME.
    LIME: Dataframe with test data.
    label_noms: Dictionary mapping labels to names.
    """
    for i in mal_classificades:
        predit = y_pred[i]
        test = y_test[y_test.index[i]]
        print("\033[1m" + "True disease: " + "\033[0m", label_noms[test])
        print("\033[1m" + "Predicted disease: " + "\033[0m", label_noms[predit])

        disease = LIME.loc[y_test.index[i]]
        exp = explainer.explain_instance(disease, predict_probaLIME, num_features=10, top_labels=num)

        # Save the explanation as an HTML file
        html_filename = f'lime_explanation({label_noms[test]})_{i}.html'
        exp.save_to_file(html_filename)

        print(f"LIME explainibility saved as {html_filename}")
        print("\n" + "-"*75 + "\n")


def evaluate_top3_accuracy_and_explain(explainer, grid_search_clf, X_test, y_test, y_pred,predict_probaLIME, LIME, label_noms):
    # Predict the top 3 classes
    p = grid_search_clf.predict_proba(X_test)
    top_classes = np.argsort(p, axis=1)[:, -3:]

    # Calculate top-3 accuracy
    correct_predictions = sum(y_bona in y_predida for y_bona, y_predida in zip(y_test, top_classes))
    accuracy = correct_predictions / len(y_test)
    print(f"Top-3 Accuracy: {accuracy}")

    # Find misclassified instances
    misclassified_indices = [i for i, (y_bona, y_predida) in enumerate(zip(y_test, top_classes)) if y_bona not in y_predida]

    for i in misclassified_indices:
        predit = y_pred[i]
        test = y_test[y_test.index[i]]
        print("\033[1m" + "True disease: " + "\033[0m", label_noms[test])
        print("\033[1m" + "Predicted disease: " + "\033[0m", label_noms[predit])

        disease = LIME.loc[y_test.index[i]]
        exp = explainer.explain_instance(disease, predict_probaLIME, num_features=10, top_labels=5)

        # Save the explanation as an HTML file
        html_filename = f'lime_explanation_TOP3({label_noms[test]}){i}.html'
        exp.save_to_file(html_filename)

        print(f"LIME explainibility TOP 3 saved as {html_filename}")
        print("\n" + "-"*75 + "\n")
    return accuracy
