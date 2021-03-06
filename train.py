import os
import json
import torch
import argparse
import numpy as np
from importlib import import_module
import sys
import time
from datetime import datetime


import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from load_data import CustomDataLoader, PseudoTrainset
from augmentation import BasicAugmentation
# from model import FCN8s


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        # mIoU_list = []
        hist = np.zeros((12, 12))
        # for step, (images, masks, _) in enumerate(data_loader):
        for step, (images, masks) in enumerate(data_loader):
    
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)

            # mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # mIoU_list.append(mIoU)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            
        avrg_loss = total_loss / cnt
        # print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}, acc: {:.4f}'.format(epoch, avrg_loss, mIoU, acc))

    return avrg_loss, mIoU


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, model_name):
    def save_model(model, saved_dir, file_name='fcn8s_best_model(pretrained).pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)
    # start training
    print('Start training..')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 9, 11, 17, 18])
    best_loss = 9999999
    best_mIoU = 0
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        # for step, (images, masks, _) in enumerate(data_loader):
        for step, (images, masks) in enumerate(data_loader):
    
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu ????????? ?????? device ??????
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss ?????? (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            # step ????????? ?????? loss ??????
            if (step + 1) % 25 == 0:
                time_taken = time.time() - start_time
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time taken: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(data_loader), loss.item(), time_taken))
                start_time = time.time()
        
        # validation ????????? ?????? loss ?????? ??? best model ??????
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            # scheduler.step(mIoU)
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('epoch: {}, lr: {}'.format(epoch + 1, param_group['lr']))
            if avrg_loss < best_loss:
                print('Best loss performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss

                # save best model
                check_point = {'net': model.state_dict()}
                output_path = os.path.join(saved_dir, f"{model_name}_best_loss.pt")
                torch.save(model.state_dict(), output_path)

            if mIoU > best_mIoU:
                print('Best mIoU performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = mIoU

                # save best model
                check_point = {'net': model.state_dict()}
                output_path = os.path.join(saved_dir, f"{model_name}_best_mIoU.pt")
                torch.save(model.state_dict(), output_path)
        
    # save last model
    output_path = os.path.join(saved_dir, f"{model_name}_last.pt")
    torch.save(model.state_dict(), output_path)


def main(args):
    # fix seed
    seed_everything(seed=1024)

    # defind save directory
    save_dir = increment_path(os.path.join(args.model_dir, args.model))
    os.makedirs(save_dir, exist_ok=True)

    # save args on .json file
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # open sys logging
    consoleLog = Logger(save_dir=save_dir)
    sys.stdout = consoleLog

    # set dataset
    dataset_path = '../data'
    anns_file_path = dataset_path + '/' + 'train.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    category_names = get_category_names(dataset)

    # train.json / validation.json
    if args.valid:
        train_path = dataset_path + '/train.json'
    else:
        train_path = dataset_path + '/train_all.json'
    val_path = dataset_path + '/val.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # set augmentation methods
    transform_module = getattr(import_module("augmentation"), args.augmentation)
    train_transform = transform_module()
    val_transform = BasicAugmentation()

    # validation set??? ?????? ????????? ?????? ??????
    # random_split ???????????? data set??? 8:2 ??? ??????
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    # train_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=train_path, category_names=category_names, mode='train', transform=train_transform)
    train_dataset = PseudoTrainset(data_dir=train_path, transform=train_transform)

    # validation dataset
    # val_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=val_path, category_names=category_names, mode='val', transform=val_transform)
    val_dataset = PseudoTrainset(data_dir=val_path, transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn,
                                            drop_last=True
                                            )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn
                                            )

    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=12).to(device)

    # define optimizer  & criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_module = getattr(import_module("loss"), args.loss)
    criterion = criterion_module()

    # train model
    val_every = 1
    train(args.epochs, model, train_loader, val_loader, criterion, optimizer, save_dir, val_every, device, args.model)

    # close sys logging
    del consoleLog

    # print current time
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=1024, help='random seed (default: 1024)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train (deafult: 20)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (deafult: 8)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader (default: 2)')
    parser.add_argument('--smoothing', type=float, default=0.2, help='label smoothing facotr for label smoothing loss (default: 0.2)')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')

    parser.add_argument('--model_dir', type=str, default='./results', help='directory where model would be saved (default: ./results)')

    # FCN8s / DeconvNet / SegNet / Deeplab_V3_Resnet101 / deeplabv3_resnet50 / Unet_resnet50 / DeepLabV3Plus_resnet101 / DeepLabV3Plus_efficientnet
    parser.add_argument('--model', type=str, default='DeepLabV3Plus_resnet101', help='backbone bert model for training (default: FCN8s)')
    # BasicAugmentation / ImagenetDefaultAugmentation / CustomAugmentation / CustomElasticAugmentation
    parser.add_argument('--augmentation', type=str, default='ImagenetDefaultAugmentation', help='augmentation method for training')
    # CrossEntropyLoss/ FocalLoss
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function to train the model')
    parser.add_argument('--valid', type=int, default=1, help='whether split training set (default: 1)')

    args = parser.parse_args()

    main(args)