import itertools 
import matplotlib.pyplot as plt
import numpy as np

# Function that creates a confusion matrix
def create_confusion_matrix(confusion_matrix, title):
    
    plt.figure(figsize = (8,5))
    classes = ['Different','Matching']
    cmap = plt.cm.Blues
    plt.grid(False)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],horizontalalignment="center",color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.ylim([1.5, -.5])
    plt.show() 