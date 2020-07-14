import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.multiclass import unique_labels


"""
IMPORTANT must upgrade Seaborn to use in google Colab.
Classification_report is just the sklearn classification report
Classification_report will show up in the shell and notebooks
Results from confusion_viz will appear in notebooks only
"""


def classification_visualization(y_true, y_pred, y_pred_prob, identifier):
    """
    Prints the results of the functions. That's it
    """
    clr = classification_report(y_true, y_pred, output_dict=True)
    y_true = np.array(y_true).ravel()
    labels = unique_labels(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"{identifier}", x=0, y=1, fontsize=16)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(f"confusion matrix".title(), loc="left")
    with sns.plotting_context(font_scale=2):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=",",
            linewidths=1,
            linecolor="grey",
            square=False,
            xticklabels=["Predicted\n" + str(i) for i in labels],
            yticklabels=["Actual\n" + str(i) for i in labels],
            robust=True,
            cmap=sns.color_palette("coolwarm"),
        )

    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    ax2 = fig.add_subplot(1, 2, 2)
    # ax2.set_title(f"Model: {identifier} decision matrix".title(),loc="center")
    ddf = pd.DataFrame(clr).T.drop(columns=["support"], axis=1)
    ax2.axis("tight")
    ax2.axis("off")
    _table = ax2.table(
        cellText=np.round(ddf.values, 2),
        loc="right",
        colLabels=ddf.columns,
        rowLabels=ddf.index,
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(16)
    _table.scale(1, 5)
    fig.tight_layout()
    plt.show()


def confusion_viz(y_true, y_pred):
    """
    Uses labels as given
    Pass y_true,y_pred, same as any sklearn classification problem
    Inspired from code from a Ryan Herr Lambda School Lecture
    """
    y_true = np.array(y_true).ravel()
    labels = unique_labels(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    sns.set(font_scale=2)
    graph = sns.heatmap(
        matrix,
        annot=True,
        fmt=",",
        linewidths=1,
        linecolor="grey",
        square=False,
        xticklabels=["Predicted\n" + str(i) for i in labels],
        yticklabels=["Actual\n" + str(i) for i in labels],
        robust=True,
        cmap=sns.color_palette("coolwarm"),
    )
    return graph


def plt_prc(y_true, y_pred):
    aps = round(average_precision_score(y_true, y_pred) * 100, 2)
    a, b, c = precision_recall_curve(y_true, y_pred)
    plt.figure()
    lw = 2
    plt.plot(a, b, color="darkorange", lw=lw, label="Precision Recall curve")
    plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Precision Recall Curve\nAverage Precision Score = {aps}%")
    plt.legend(loc="lower right")
    plt.show()


def plt_roc(y_true, y_pred):
    plt.figure()
    lw = 2
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    ras = round(roc_auc_score(y_true, y_pred) * 100, 2)
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve ")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC AUC Curve\nROC AUC Score = {ras}%")
    plt.legend(loc="lower right")
    plt.show()
