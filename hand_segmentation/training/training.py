import torch
import numpy as np
from tqdm import tqdm
from metrics import pixelAcc
from metrics import meanIOU


def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader, n_ensemble, model_store_path=None, lastCkptPath=None):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    tr_loss_arr = []
    val_loss_arr = []
    meanioutrain = []
    pixelacctrain = []
    meanioutest = []
    pixelacctest = []
    prevEpoch = 0

    if lastCkptPath is not None:
        checkpoint = torch.load(lastCkptPath)
        prevEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    tr_loss_arr = checkpoint['Training Loss']
        val_loss_arr = checkpoint['Validation Loss']
        meanioutrain = checkpoint['MeanIOU train']
        pixelacctrain = checkpoint['PixelAcc train']
        meanioutest = checkpoint['MeanIOU test']
        pixelacctest = checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prevEpoch)
        model.to(device)

    for epoch in range(0, n_epochs):
        train_loss = 0.0
        pixelacc = 0
        meaniou = 0

        # TRAINING
        pbar = tqdm(train_loader, total=len(train_loader))
        for X, y in pbar:
            torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)
            loss = loss_fn(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss_arr.append(loss.item())
            meanioutrain.append(meanIOU(y, ypred))
            pixelacctrain.append(pixelAcc(y, ypred))
            pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                              'Training Loss': np.mean(tr_loss_arr),
                              'Mean IOU': np.mean(meanioutrain),
                              'Pixel Acc': np.mean(pixelacctrain)
                              })

        # VALIDATION
        with torch.no_grad():

            val_loss = 0
            pbar = tqdm(val_loader, total=len(val_loader))
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                model.eval()
                ypred = model(X)

                val_loss_arr.append(loss_fn(ypred, y).item())
                pixelacctest.append(pixelAcc(y, ypred))
                meanioutest.append(meanIOU(y, ypred))

                pbar.set_postfix({'Epoch': epoch + 1 + prevEpoch,
                                  'Validation Loss': np.mean(val_loss_arr),
                                  'Mean IOU': np.mean(meanioutest),
                                  'Pixel Acc': np.mean(pixelacctest)
                                  })

        checkpoint = {
            'epoch': epoch + 1 + prevEpoch,
            'description': "add your description",
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training Loss': tr_loss_arr,
            'Validation Loss': val_loss_arr,
            'MeanIOU train': meanioutrain,
            'PixelAcc train': pixelacctrain,
            'MeanIOU test': meanioutest,
            'PixelAcc test': pixelacctest
        }
        # torch.save(checkpoint, 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test1' + str(epoch + 1 + prevEpoch) + '.pt')
        if model_store_path is not None:
            torch.save(checkpoint, model_store_path + '/model' + str(1 + n_ensemble) + '.pt')
        lr_scheduler.step()

    return tr_loss_arr, val_loss_arr, meanioutrain, pixelacctrain, meanioutest, pixelacctest

