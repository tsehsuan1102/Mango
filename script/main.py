from argparse import ArgumentParser
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset, collate_picture



import os
import logging
import datetime

from transform import Resize
from model import MyModel
from vgg_model import get_vgg_model
from tqdm import tqdm


def train(args, model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)
    print(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for i_epoch in range(args.epoch):
        
        pbar = tqdm(train_loader)
        pbar.close()
        pbar = tqdm(train_loader)

        running_loss = 0.0

        for i, data in enumerate(pbar):
            optimizer.zero_grad()

            batch_sz = len(data['imgs'])
            #print('batch:', batch_sz)
            ### input/target
            imgs = torch.stack(data['imgs'], 0).to(device)
            #imgs = imgs.permute(0, 2, 3, 1)
            tags = torch.stack(data['tags'], 0).to(device)

            #print(imgs.shape)

            y = model(imgs)
            #print('y:', y)
            #print('tag', tags)
            loss = criterion(y, tags)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%100 == 99:
                print('[%d %5d] loss:%03f' % (i_epoch, i, running_loss/100))
                running_loss = 0.0
        torch.save(model.state_dict(), '../model/model_%s' % (str(i_epoch)))










def predict(args, model, predict_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    print(model)

    
    cnt_total = 0
    cnt_same = 0


    ## output
    output_file = open(args.output, 'w')
    


    id2tag = {'0':'A', '1':'B', '2':'C'}

    pbar = tqdm(predict_loader)
    for i, data in enumerate(pbar):
        batch_sz = len(data['imgs'])
        imgs = torch.stack(data['imgs'], 0).to(device)
        #imgs = imgs.permute(0, 2, 3, 1)
        tags = torch.stack(data['tags'], 0).to(device)
        names = data['names']


        with torch.no_grad():
            y = model(imgs)
        
        print('y:', y)
        for i_ans in range(batch_sz):
            cnt_total += 1
            now_pred = y[i_ans].topk(1)[1].item()

            print('ans:', tags[i_ans].item(), ' pre:', now_pred)

            output_file.write(names[i_ans]+','+id2tag[str(now_pred)]+'\n')

            if tags[i_ans].item() == y[i_ans].topk(1)[1].item():
                cnt_same += 1
            




    print('total:', cnt_total, 'same:', cnt_same)





def get_train_transform(size): #mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        #RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1),
    ])
    return train_transform


def main(args):
    #model = MyModel(args.size)
    model = get_vgg_model()
    print(model)

    ## TODO: Crop method
    transformer = get_train_transform(512)

    if args.do_train:
        trainset = MyDataset(
            args.data_list,
            args.data_dir,
            transform = transformer
        )

        #print(trainset[1])
        #print(len(trainset))
        train_loader = DataLoader(
            dataset = trainset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = collate_picture
        )

        train(args, model, train_loader)
    
    #### predict
    if args.do_predict:
        if args.load_model == None and not args.do_train:
            logging.error('load model error')
            exit(1)

        elif args.load_model:
            logging.info('loading model...... %s' % (args.load_model))
            model.load_state_dict(torch.load(args.load_model))

        if args.predict_file == None:
            logging.error('No predict file')
            exit(1)


        logging.info('start reading predict files')
        predictset = MyDataset(
            args.predict_file,
            args.data_dir,
            transform = transformer
        )
        print(predictset[0][0].shape)
        print('len', len(predictset))

        predict_loader = DataLoader(
            dataset = predictset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn = collate_picture
        )
        predict(args, model, predict_loader)





def parse_argument():
    parser = ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='data directory')
    parser.add_argument('--data_dir', default='../data/', type=str, help='data directory')
    parser.add_argument('--data_list', default='../data/train.csv', type=str, help='data directory')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=4, type=int, help='epoch')
    parser.add_argument('--size', default=512, type=int, help='image size')
    parser.add_argument('--do_predict', action='store_true', help='evaluate')
    parser.add_argument('--load_model', type=str, help='trained model')
    parser.add_argument('--predict_file', default='../data/dev.csv', type=str, help='data directory')
    parser.add_argument('--output', default='./prediction.csv', type=str, help='my prediction')

    args = parser.parse_args()
    return args




if __name__=='__main__':
    args = parse_argument()

    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=loglevel,
        datefmt='%m-%d %H:%M:%S'
    )
    main(args)





