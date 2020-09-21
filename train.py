import os
import sys
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from core import BATCH_SIZE, SAVE_FREQ, LR, resume, save_dir
from core import model, dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

trainset = dataset.CUB(root="./CUB_200_2011", is_train=True, data_len=None)
trainloader = troch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

testset = dataset.CUB(root="./CUB_200_2011", is_train=False, data_len=None)
testloader = troch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)


net = core.resnet50(pretrained=True)
if resume:
    ckpt = torch.load(resume)
    net.load_State_dict(ckpt["net_state_dict"])
    start_epoch = ckpt["epoch"] + 1
creterion = torch.nn.CrossEntropyLoss()

net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, 10):

    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        
        raw_logits, _, _ = net(img)
        raw_loss = creterion(raw_logits, label)
        total_loss = raw_logic
        total_loss.backward()

        progress_bar(i, len(trainloader), "train")

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                raw_logits, _, _ = net(img)
                raw_loss = creterion(raw_logits, label)
                _, raw_predict = torch.max(raw_logits, 1)
                total += batch_size
                train_correct += raw_loss.item() * batch_size
                progress_bar(i, len(trainloader), "eval train set")
        
        train_acc = float(train_correct) / total
        train_loss = tran_loss / total

        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                raw_logits, _, _ net(img)
                raw_loss = creterion(raw_logits, label)
                _, raw_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(raw_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), "eval test set")
        
        test_acc = float(test_correct) / total