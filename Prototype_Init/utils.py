import numpy as np
import random
import io
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
# from skimage import color
from sklearn import metrics
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from captum.attr import InputXGradient, IntegratedGradients, DeepLift, NoiseTunnel

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def create_writer(args):
    writer = SummaryWriter(f"runs/{args.conf_version}/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    # print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {:.3f} Recall: {:.3f}, Accuracy: {:.3f}: ,f1_score: {:.3f}'.format(precision*100,recall*100,
                                                                                         accuracy*100,f1_score*100))
    return precision, recall, accuracy, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None,
                          cmap=plt.cm.Blues, sFigName='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
# Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(sFigName)
    return ax


def resize_tensor(input_tensors, h, w):
    input_tensors = torch.squeeze(input_tensors, 1)

    for i, img in enumerate(input_tensors):
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if i == 0:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
    return final_output
