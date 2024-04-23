from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt


def sensitivity_and_specificity(confusion_matrix):
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def acc(confusion_matrix):
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    return accuracy

def roc_plot(model, X_num, X_encoded, Y):
    y_pred = model.predict([X_num, X_encoded])

    # Compute ROC curve and AUC for training data
    fpr, tpr, _ = roc_curve(Y, y_pred)
    roc_auc = roc_auc_score(Y, y_pred)
    
    return fpr, tpr, roc_auc

    
def confusion_matrix_display(y_pred, Y, title, threshold=0.5): 
    y_pred_classes = (y_pred > threshold).astype(int)

    # Get the confusion matrix
    cm = confusion_matrix(Y, y_pred_classes, normalize='all')  # Normalizing by all values

    # Define class names (if applicable)
    class_names = ['0', '1']  # Modify this according to your classes


    # Display the confusion matrix with relative frequencies
    accuracy = acc(cm)
    print("\n", title)
    print(f"Accuracy: {round(accuracy*100, 2)}%")
    sensitivity, specificity = sensitivity_and_specificity(cm)
    print(f'Sensitivity: {round(sensitivity, 3)}')
    print(f'Specificity: {round(specificity, 3)}')
    auc = roc_auc_score(Y, y_pred)
    print(f'AUC: {round(auc, 4)}')


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
