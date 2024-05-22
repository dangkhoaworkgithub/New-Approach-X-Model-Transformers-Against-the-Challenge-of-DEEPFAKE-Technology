import torch
import sys, os
import argparse
import numpy as np
import seaborn as sns
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import time
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report

sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')
sys.path.insert(1,'preprocessing')

from augmentation import Aug
from xmodel import XMT
from loader import session, curriculum_loader
import optparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Default
cession='g' # GPU runtime
epoch = 1
dir_path = ""
batch_size = 32
lr=0.0001
weight_decay=0.0000001

parser = optparse.OptionParser("Train XMT model.")
parser.add_option("-e", "--epoch", type=int, dest='epoch', help='Number of epochs used for training the X model.')
parser.add_option("-v", "--version", dest='version', help='Version 0.1.')
parser.add_option("-s", "--cession", type="string",dest='session', help='Training session. Use g for GPU, t for TPU.')
parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
parser.add_option("-l", "--rate",  type=float, dest='rate', help='Learning rate.')
parser.add_option("-w", "--decay", type=float, dest='decay', help='Weight decay.')
parser.add_option("-p", "--plot", action="store_true", dest='plot', help='Plot training and validation metrics.')
parser.add_option("-u", "--curriculum_level", type="string", dest="curriculum_level",
                  help="Specify the difficulty level for curriculum learning: easy, medium, hard, or leave blank for normal dataset structure.")

(options,args) = parser.parse_args()

plot_metrics = False

if options.session:
    cession = options.session
if options.dir==None:
    print (parser.usage)
    exit(0)
else:
    dir_path = options.dir
if options.batch:
    batch_size = int(options.batch)
if options.epoch:
    epoch = int(options.epoch)
if options.rate:
    lr = float(options.rate)
if options.decay:
    weight_decay = float(options.decay)
if options.plot:
    plot_metrics = True

if options.curriculum_level:
    curriculum_level = options.curriculum_level.lower()
    if curriculum_level in ['easy', 'medium', 'hard']:
        dataloaders, dataset_sizes = curriculum_loader(cession, dir_path, batch_size, curriculum_level)
        print(f"Loaded curriculum data for level: {curriculum_level}")
    else:
        print("Invalid curriculum level specified. Choose 'easy', 'medium', 'hard'.")
        sys.exit(1)
else:
    batch_size, dataloaders, dataset_sizes = session(cession, dir_path, batch_size)


#X model definition
model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
num_epochs = epoch
min_val_loss=10000
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

def log_record_g(metrics, filename):
    writeable_dir = 'result/'
    filepath = writeable_dir + filename

    with open(filepath, "a") as file:
        for key, values in metrics.items():
            file.write(f"{key}: {values[-1]}\n")
        file.write("\n")

def log_record_c(metrics, filename):
    writeable_dir = 'result/'
    filepath = writeable_dir + filename

    with open(filepath, "a") as file:
        for key, values in metrics.items():
            if values:  # Check if values list is not empty
                file.write(f"{key}: {values[-1]}\n")
        file.write("\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_cpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics):

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []
    test_loss = []
    test_acc = []

    metrics = {
        'train_loss': train_loss,
        'train_acc': train_accu,
        'val_loss': val_loss,
        'val_acc': val_accu,
        'test_loss': test_loss,
        'test_acc': test_acc
    }

    model_path = 'weight/xmodel_deepfake_sample_1.pth'
    if os.path.exists(model_path):
        print("Loading saved model...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_loss']
    else:
        print("Train from begining...")
        start_epoch = 0

    since = time.time()

    best_model_saved = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    epoch_loss = None
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_idx=0

            epoch_loss = running_loss / dataset_sizes[phase]

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  # Corrected line                

                if phase_idx%100==0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, phase_idx * batch_size, dataset_sizes[phase],\
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < min_loss and epoch_loss is not None:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Validation loss decreased ({:.6f}). Saving model ...'.format(min_loss))

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': best_model_wts,
                    'optimizer': optimizer.state_dict(),
                    'min_loss': min_loss
                }, 'weight/xmodel_deepfake_sample_1.pth')
                best_model_saved = True

    test_loss_value, test_acc_value = test(model, dataloaders, dataset_sizes, device, criterion)
    test_loss.append(test_loss_value)
    test_acc.append(test_acc_value)

    log_record_c(metrics, 'log_records.txt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    with open('weight/xmodel_deepfake_sample_1.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    if not best_model_saved:
        print("Best model was not updated in the last epochs, saving last model state...")
        final_state = {
            'epoch': total_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': epoch_loss
        }
        torch.save(final_state, 'weight/xmodel_deepfake_sample_1.pth')

    auc_score = calculate_auc(model, dataloaders, dataset_sizes)
    print('AUC:', auc_score)

    cm = calculate_confusion_matrix(model, dataloaders, dataset_sizes)
    print('confusion_matrix', cm)

    if plot_metrics == True:
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/training_validation_loss.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy")
        plt.plot([acc.cpu().numpy() for acc in train_accu], label="Train Accuracy")
        plt.plot([acc.cpu().numpy() for acc in val_accu], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/training_validation_accuracy.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Test Loss")
        plt.plot(test_loss, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/test_loss.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Test Accuracy")
        plt.plot([acc.cpu().numpy() for acc in test_acc], label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/test_accuracy.png')
        plt.close()

    return train_loss,train_accu,val_loss,val_accu, min_loss

def train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics):

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []
    test_loss = []
    test_acc = []

    metrics = {
        'train_loss': train_loss,
        'train_acc': train_accu,
        'val_loss': val_loss,
        'val_acc': val_accu,
        'test_loss': test_loss,
        'test_acc': test_acc
    }

    model_path = 'weight/xmodel_deepfake_sample_1.pth'
    if os.path.exists(model_path):
        print("Loading saved model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_loss']
    else:
        start_epoch = 0

    since = time.time()

    best_model_saved = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_loss = min_val_loss
    epoch_loss = None
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        print('Epoch {}/{}'.format(epoch, total_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_idx = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase_idx%100==0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, phase_idx * batch_size, dataset_sizes[phase], \
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < min_loss and epoch_loss is not None:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Validation loss decreased ({:.6f}). Saving model ...'.format(min_loss))

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': best_model_wts,
                    'optimizer': optimizer.state_dict(),
                    'min_loss': min_loss
                }, 'weight/xmodel_deepfake_sample_1.pth')
                best_model_saved = True

        test_loss_value, test_acc_value = test(model, dataloaders, dataset_sizes, device, criterion)
        test_loss.append(test_loss_value)
        test_acc.append(test_acc_value)

        log_record_g(metrics, 'log_records.txt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    with open('weight/xmodel_deepfake_sample_1.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu, test_loss, test_acc], f)

    if not best_model_saved:
        print("Best model was not updated in the last epochs, saving last model state...")
        final_state = {
            'epoch': total_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': epoch_loss
        }
        torch.save(final_state, 'weight/xmodel_deepfake_sample_1.pth')

    auc_score = calculate_auc(model, dataloaders, dataset_sizes)
    print('AUC:', auc_score)

    cm = calculate_confusion_matrix(model, dataloaders, dataset_sizes)
    print('confusion_matrix', cm)

    if plot_metrics:
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/training_validation_loss.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy")
        plt.plot([acc.cpu().numpy() for acc in train_accu], label="Train Accuracy")
        plt.plot([acc.cpu().numpy() for acc in val_accu], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/training_validation_accuracy.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Test Loss")
        plt.plot(test_loss, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/test_loss.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Test Accuracy")
        plt.plot([acc.cpu().numpy() for acc in test_acc], label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/test_accuracy.png')
        plt.close()

    return train_loss, train_accu, val_loss, val_accu, min_loss

def calculate_auc(model, dataloaders, dataset_sizes):
    model.eval()
    all_labels = []
    all_preds = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        outputs = outputs.softmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(outputs[:, 1].tolist())

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('result/roc_curve.png')
    plt.close()

    return roc_auc

def calculate_confusion_matrix(model, dataloaders, dataset_sizes):
    model.eval()
    all_labels = []
    all_preds = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.tolist())
        all_preds.extend(predictions.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    f1_score = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    print(f"F1 Score: {f1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.title('Confusion Matrix')
    plt.savefig('result/confusion_matrix.png')
    plt.close()

    return cm

def test(model, dataloaders, dataset_sizes, device, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / dataset_sizes['test']
    total_acc = running_corrects.double() / dataset_sizes['test']
    accuracy_percentage = total_acc * 100

    print('Test Loss: {:.4f}'.format(total_loss))
    print('Test Accuracy: {:.2f}%'.format(accuracy_percentage))

    return total_loss, accuracy_percentage


if cession == 'c':
    train_cpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics)
else:
    train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics)
