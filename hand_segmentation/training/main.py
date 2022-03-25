import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from torch import optim, nn
from torch.utils.data import ConcatDataset, DataLoader
from fetchmodel import HandSegModel
from segds import SegDataset, trainTestSplit
from training import training_loop

if __name__ == '__main__':
    M = 1   # number of ensemble members
    N_epochs = 3
    batchSize = 2
    dataSet = -1
    learning_rate = 0.0002
    model_store_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/models/11'
    dset = 11

    # Write parameters in text file
    file = open(model_store_path + '/params.txt', 'w')
    file.write('M = ' + str(M) + ', N_epochs = ' + str(N_epochs) + ', batchSize = ' + str(batchSize) + ', dataSet = '
               + str(dataSet) + ', learning rate = ' + str(
        learning_rate) + ', dataset = paper/dset' + str(dset))

    file.close()

    HOFdataset = SegDataset('C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face', 'images_resized',
                            'masks')
    Owndataset = SegDataset('C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets', 'images/dset' + str(dset),
                            'masks/dset' + str(dset))

    # Pick correct dataset
    if dataSet == -1:
        megaDataset = Owndataset
    elif dataSet == 0:
        megaDataset = HOFdataset
    elif dataSet == 1:
        megaDataset = ConcatDataset([HOFdataset, Owndataset])
    elif dataSet == 2:
        GTEAdataset = SegDataset('C:/Users/ralf-/Documents/Python/SemanticSegmentation/GTEAhands', 'Images', 'Masks')
        megaDataset = ConcatDataset([HOFdataset, GTEAdataset])
    else:
        GTEAdataset = SegDataset('C:/Users/ralf-/Documents/Python/SemanticSegmentation/GTEAhands', 'Images', 'Masks')
        EGOdataset = SegDataset('egodata', 'test_images', 'masks')
        megaDataset = ConcatDataset([EGOdataset, HOFdataset, GTEAdataset])

    trainDataset, valDataset = trainTestSplit(megaDataset, 0.99)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, drop_last=True)  # data is reshuffled at every epoch
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, drop_last=True)

    for n_ensemble in range(M):
        model = HandSegModel()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
        retval = training_loop(N_epochs,
                               optimizer,
                               lr_scheduler,
                               model,
                               loss_fn,
                               trainLoader,
                               valLoader,
                               n_ensemble,
                               model_store_path)
                               # 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/checkpoints/checkpointhandseg1.pt')


# after the training loop returns, we can plot the data
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
N = 10
ax[0][0].plot(running_mean(retval[0], N), 'r.', label='training loss')
ax[1][0].plot(running_mean(retval[1], N), 'b.', label='validation loss')
ax[0][1].plot(running_mean(retval[2], N), 'g.', label='meanIOU training')
ax[1][1].plot(running_mean(retval[4], N), 'r.', label='meanIOU validation')
ax[0][2].plot(running_mean(retval[3], N), 'b.', label='pixelAcc training')
ax[1][2].plot(running_mean(retval[5], N), 'b.', label='pixelAcc validation')
for i in ax:
    for j in i:
        j.legend()
        j.grid(True)
plt.show()
