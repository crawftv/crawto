import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
"""
IMPORTANT must upgrade Seaborn to use in google Colab.
Classification_report is just the sklearn classification report
Classification_report will show up in the shell and notebooks
Results from confusion_viz will appear in notebooks only
"""
def classification_visualization(y_true,y_pred):
    """
    Prints the results of the functions. That's it
    """
    fig = plt.figure(figsize=(12,6))
    fig.tight_layout()
    label1 =classification_report(y_true,y_pred)
    print(label1)
    plt.figure(1)
    confusion_viz(y_true,y_pred)
    plt.figure(2)
    plt_prc(y_true,y_pred)
    
    plt.show();

def confusion_viz(y_true, y_pred):
    """
    Uses labels as given
    Pass y_true,y_pred, same as any sklearn classification problem
    Inspired from code from a Ryan Herr Lambda School Lecture
    """

    y_true = np.array(y_true).ravel()
    labels = unique_labels(y_true,y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    graph = sns.heatmap(matrix, annot=True,
                       fmt=',', linewidths=1,linecolor='grey',
                       square=True,
                       xticklabels=["Predicted\n" + str(i) for i in labels],
                       yticklabels=["Actual\n" + str(i) for i in labels],
                       robust=True,
                       cmap=sns.color_palette("coolwarm"))
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    return graph


def plt_prc(y_true, y_pred):
    a,b,c  = precision_recall_curve(y_true,y_pred)
    plt.figure()
    lw = 2
    plt.plot(a, b, color='darkorange',
             lw=lw, label='Precision Recall curve')#(area = %0.2f)' % c)
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt