from DatasetTrafficSign import Traffic_sign_dataset
import torch
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToPILImage, ToTensor, Compose, Normalize, Resize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from SiameseNetwork_scratch import Siamese
from ContrastiveLoss import ContrastiveLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from Data4Knn import DataForKnn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

if __name__ == '__main__':
    """Check cuda is available or not"""
    if torch.cuda.is_available():
       device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    """Declare transform"""
    transform = Compose([
        ToPILImage(),
        Resize((105, 105)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])

    """Dataset for train Siamese and KNN"""
    train_dataset = Traffic_sign_dataset(root='', transform=transform, is_train=True)
    valid_dataset = Traffic_sign_dataset(root='', transform=transform, is_train=False)
    databaseSet4knn   = DataForKnn(root='', transform=transform, databaseSet=True)
    datavalSet4knn    = DataForKnn(root='', transform=transform, databaseSet=False)

    example_data_input = train_dataset.__getitem__(index=0) #(image1, image2), label
    ex_img1, ex_img2 = example_data_input[0]
    ex_input1 = ex_img1.unsqueeze(0).to(device)
    ex_input2 = ex_img2.unsqueeze(0).to(device)


    """DataLoader for Siamese and KNN"""
    train_dataloader = DataLoader(train_dataset, batch_size=128, drop_last=False, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, drop_last=False, shuffle=False)
    KnnBase_dataloader   = DataLoader(databaseSet4knn, batch_size=64, drop_last=False, shuffle=True)
    KnnValid_dataloader  = DataLoader(datavalSet4knn, batch_size=64, drop_last=False, shuffle=True)

    """Declare model"""
    siameseModel = Siamese()
    siameseModel.to(device)
    siameseModel = nn.DataParallel(siameseModel)

    """Declare loss function"""
    loss_function = ContrastiveLoss(margin=1.0).to(device)
    """Declare optimizer"""
    optimzer = SGD(siameseModel.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    """Change lr"""
    scheduler = ReduceLROnPlateau(optimizer=optimzer, factor=0.1, patience=20, mode='min')

    """Tracking and Visualize"""
    writer = SummaryWriter(log_dir='')
    writer.add_graph(model = siameseModel, input_to_model = [ex_input1, ex_input2])

    """Training process"""
    epochs=200
    train_global_step = 0
    valid_global_step = 0
    best_accuracy = 0
    colors = sns.color_palette("hsv", 44)
    CLASS_NAMES = databaseSet4knn.CLASS_NAMES
    number_of_class = len(CLASS_NAMES)

    for epoch in range(1, epochs+1):

        progress_bar_train = tqdm(train_dataloader)
        progress_bar_valid = tqdm(valid_dataloader)
        progress_bar_BaseKnn   = tqdm(KnnBase_dataloader)
        progress_bar_ValidKnn  = tqdm(KnnValid_dataloader)

        """Train Siamese"""
        siameseModel.train()
        for iter, ((image1, image2), label) in enumerate(progress_bar_train):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label  = label.to(device)

            #forward
            output1, output2 = siameseModel(image1, image2)
            loss_train = loss_function(output1, output2, label)

            #backward
            optimzer.zero_grad()
            loss_train.back_ward()
            optimzer.step()
            # scheduler.step()

            writer.add_scalar(tag= 'LossTrain', scalar_value= loss_train, global_step=train_global_step)
            progress_bar_train.set_description(f'Train Siamese Phase. Epoch: {epoch}, iteration: {iter}, loss: {loss_train}.')
            train_global_step = train_global_step + 1

        """Assess Siamese"""
        running_valid_loss = 0.0
        siameseModel.eval()
        with torch.inference_mode():
            for iter, ((image1, image2), label) in enumerate(progress_bar_valid):
                image1 = image1.to(device)
                image2 = image2.to(device)
                label  = label.to(device)

                output1, output2 = siameseModel(image1, image2)
                loss_valid = loss_function(output1, output2, label)

                running_valid_loss += loss_valid.item()

                writer.add_scalar(tag= 'LossValid', scalar_value= loss_valid, global_step=valid_global_step)
                progress_bar_valid.set_description(f'Valid Siamese Phase. Epoch: {epoch}, iteration: {iter}, loss: {loss_valid}.')
                valid_global_step = valid_global_step + 1

        avg_valid_loss = running_valid_loss / len(valid_dataloader)
        scheduler.step(avg_valid_loss)

        knn = KNeighborsClassifier(n_neighbors=5)

        """ADD Data into KNN"""
        siameseModel.eval()
        vector_store = []
        label_store  = []
        with torch.inference_mode():
            for iter,(images, labels) in enumerate(progress_bar_BaseKnn):
                images = images.to(device)

                embed_vector = siameseModel.module.forward_once(images) # batch,dim
                vector_store.extend(embed_vector.cpu().numpy())
                label_store.extend(labels.numpy())

        knn.fit(X=np.array(vector_store), y=np.array(label_store))

        """Assess KNN"""
        siameseModel.eval()
        actual_label = []
        vector_store = []

        with torch.inference_mode():
            for iter,(images, labels) in enumerate(progress_bar_ValidKnn):
                images = images.to(device)

                embed_vector = siameseModel.module.forward_once(images)  # batch,dim
                vector_store.extend(embed_vector.cpu().numpy())
                actual_label.extend(labels.numpy())

        vector_store = np.array(vector_store)

        predict_label = knn.predict(X=vector_store)
        actual_label = np.array(actual_label)

        metric = classification_report(y_true=actual_label, y_pred=predict_label, output_dict=True)
        accuracy = metric['accuracy']
        writer.add_scalar('Accuracy/KNN_Validation', accuracy, epoch)

        if accuracy > best_accuracy:
            torch.save(siameseModel.state_dict(), f'SiameseModel_Traffic_Best.pt')
            best_accuracy = accuracy

        if epoch in [1, 50, 100, 150, 200]:
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(vector_store)

            fig = plt.figure(figsize=(12, 9))
            ax = plt.gca()

            for v, label in zip(emb_2d, actual_label):
                fig=sns.scatterplot(x=[v[0]], y=[v[1]], color=colors[label],
                                palette = colors, legend=False, s=30, ax = ax)

            handles = [
                mpatches.Patch(color=colors[i], label=f"{CLASS_NAMES[i]}")
                for i in range(number_of_class)
            ]

            plt.legend(
                handles=handles,
                title="Classes",
                bbox_to_anchor=(1.05, 1.05),
                loc="upper left",
                borderaxespad=0.,
                fontsize=7
            )

            plt.subplots_adjust(
                left=0.08,
                right=0.72,
                top=0.95,
                bottom=0.1,
                wspace=0.3,
                hspace=0.3
            )

            writer.add_figure(tag=f'Position datapoints in epoch {epoch}', figure=fig)
            plt.close(fig)

    writer.close()