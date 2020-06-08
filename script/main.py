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













def evaluate(answers, predictions):
    # print('answer: ', answers)
    # print('predict: ', predictions)
    count_ans = {'A':0, 'B':0, 'C':0}
    count_pred = {'A':0, 'B':0, 'C':0}
    acc = {'A':0, 'B':0, 'C':0}

    for key in answers.keys():
        count_ans[answers[key]] += 1
        count_pred[predictions[key]] += 1

        if answers[key] == predictions[key]:
            acc[answers[key]] += 1


    print('ans: ', count_ans)
    print('prediction: ', count_pred)
    print('acc', acc)
    print('recallA:', acc['A']/count_ans['A'])
    print('recallB:', acc['B']/count_ans['B'])
    print('recallC:', acc['C']/count_ans['C'])
        
    weight = {'A':0, 'B':0, 'C':0}
    total_num = count_ans['A'] + count_ans['B'] + count_ans['C']
    for k in ['A', 'B', 'C']:
        weight[k] = count_ans[k] / total_num
    # print(weight)

    WAR = 0.0
    for k in ['A', 'B', 'C']:
        ### weight * recall
        WAR += weight[k] * (acc[k]/count_ans[k])
    print(WAR)
    return WAR




def train(args, model, train_loader, dev_loader, start_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2tag = {'0':'A', '1':'B', '2':'C'}
    
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


    for i_epoch in range(args.epoch):
        pbar = tqdm(train_loader)
        running_loss = 0.0
    
        logging.info('[Train]')    
        
        for i, data in enumerate(pbar):
            optimizer.zero_grad()

            batch_sz = len(data['imgs'])
            #print('batch:', batch_sz)
            ### input/target
            imgs = torch.stack(data['imgs'], 0).to(device)
            #imgs = imgs.permute(0, 2, 3, 1)
            tags = torch.stack(data['tags'], 0).to(device)

            y = model(imgs)
            
            loss = criterion(y, tags)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #if i%100 == 99:
            #    print('[%d %5d] loss:%03f' % (i_epoch, i, running_loss/100))
            #    running_loss = 0.0
        ## finish a epoch
        ## save model to specific directory
        

    ### run on dev set to avoid overfitting
        logging.info('[Dev]')
        dev_pbar = tqdm(dev_loader)
        ### id->label
        answers = {}
        predictions = {}
        for i, data in enumerate(dev_pbar):
            batch_sz = len(data['imgs'])
            imgs = torch.stack(data['imgs'], 0).to(device)
            tags = torch.stack(data['tags'], 0).to(device)
            names = data['names']
            #print(names)
            ### with no grad
            with torch.no_grad():
                y = model(imgs)
            
            for i_ans in range(batch_sz):
                now_pred = y[i_ans].topk(1)[1].item()
                predictions[names[i_ans]] = id2tag[str(now_pred)]
                answers[names[i_ans]] = id2tag[str(tags[i_ans].item())]
        
        WAR = evaluate(answers, predictions)

        torch.save({
                'epoch': i_epoch + start_epoch,
                'state_dict': model.state_dict(),
                'WAR': WAR,
            }, '../new_model/model_%s' % (str(i_epoch + start_epoch))
        )




def predict(args, model, predict_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

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
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomCrop(size),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1),
    ])
    return train_transform


def get_predict_transform(size): #mean=mean, std=std, size=0):
    predict_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1),
    ])
    return predict_transform




def main(args):
    #model = MyModel(args.size)
    model = get_vgg_model()
    print(model)

    ## TODO: Crop method
    if args.do_train:
        transformer = get_train_transform(512)
        dev_transformer = get_predict_transform(512)



        start_epoch = 0
        if args.load_model:
            logging.info('loading model...... %s' % (args.load_model))
            checkpoint = torch.load(args.load_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1



        ### trainset
        trainset = MyDataset(
            args.data_list,
            args.data_dir,
            transform = transformer
        )
        train_loader = DataLoader(
            dataset = trainset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = collate_picture
        )

        ### dev set
        devset = MyDataset(
            args.dev_file,
            args.data_dir,
            transform = dev_transformer
        )
        dev_loader = DataLoader(
            dataset = devset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn = collate_picture
        )
        train(args, model, train_loader, dev_loader, start_epoch)
    

    #### predict
    if args.do_predict:
        transformer = get_predict_transform(512)
        if args.load_model == None and not args.do_train:
            logging.error('load model error')
            exit(1)

        elif args.load_model:
            logging.info('loading model...... %s' % (args.load_model))
            
            checkpoint = torch.load(args.load_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            #model.load_state_dict(torch.load(args.load_model))

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
    parser.add_argument('--dev_file', default='../data/dev.csv', type=str, help='dev file')
    parser.add_argument('--predict_file', default='../data/dev.csv', type=str, help='the input file for predict')
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





