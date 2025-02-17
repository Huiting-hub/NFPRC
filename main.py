import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import argparse
import numpy as np
import datetime
import data_load
import resnet
from lenet import LeNet
import tools
import random
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='output/result/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[symmetric, asymmetric, pairflip, instance]', default='symmetric')
parser.add_argument('--dataset', type=str, help='[mnist, fmnist, cifar10, cifar100]', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=350)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--weight_decay', type=float, help='l2', default=0.001)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
parser.add_argument('--model_path', type =str, help='dir to load pretrained model', default= 'model_beat.pth')
parser.add_argument('--Lambda1', type=float, help='hyper-parameter lambda', default=0.4) 
parser.add_argument('--Lambda2', type=float, help='hyper-parameter lambda', default=0.01)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.cuda.set_device(args.gpu)

learning_rate = args.lr

# Load dataset
def load_data(args):

    if args.dataset == 'fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 50
        args.batch_size = 32
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.fmnist_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        
        feat_dataset = train_dataset

        val_dataset = data_load.fmnist_dataset(False,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        test_dataset = data_load.fmnist_test_dataset(
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                target_transform=tools.transform_target)

    if args.dataset == 'mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 50
        args.batch_size = 32
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.mnist_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)),]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        
        feat_dataset = train_dataset

        val_dataset = data_load.mnist_dataset(False,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)),]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        test_dataset = data_load.mnist_test_dataset(
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)),]),
                                                target_transform=tools.transform_target)

    if args.dataset == 'cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 50
        args.batch_size = 64
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar10_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        
        feat_dataset = data_load.cifar10_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        val_dataset = data_load.cifar10_dataset(False,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        test_dataset = data_load.cifar10_test_dataset(
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target)

    if args.dataset == 'cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 50
        args.batch_size = 64
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar100_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)
        
        feat_dataset = data_load.cifar100_dataset(True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target,
                                                dataset=args.dataset,
                                                noise_type=args.noise_type,
                                                noise_rate=args.noise_rate,
                                                split_per=args.split_percentage,
                                                random_seed=args.seed)

        val_dataset = data_load.cifar100_dataset(False,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                 target_transform=tools.transform_target,
                                                 dataset=args.dataset,
                                                 noise_type=args.noise_type,
                                                 noise_rate=args.noise_rate,
                                                 split_per=args.split_percentage,
                                                 random_seed=args.seed)

        test_dataset = data_load.cifar100_test_dataset(
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                ]),
                                                target_transform=tools.transform_target)

    return train_dataset, feat_dataset, val_dataset, test_dataset

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def feature_extractor(model_path, model1, feat_loader):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    state_dict = {k[len('encoder_q.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder_q.')}

    # mnist
    # state_dict.pop('fc3.0.weight')
    # state_dict.pop('fc3.0.bias')
    # state_dict.pop('fc3.2.weight')
    # state_dict.pop('fc3.2.bias')

    # other datasets
    state_dict.pop('linear.0.weight')
    state_dict.pop('linear.0.bias')
    state_dict.pop('linear.2.weight')
    state_dict.pop('linear.2.bias')
    model1.load_state_dict(state_dict, strict=False)
    print("encoder_q loaded successfully from", model_path)

    model1 = model1.cuda()
    model1.eval()
    feature_bank=[]
    with torch.no_grad():
        for i, (data, target, indexes) in  enumerate(feat_loader):
            feature, _ = model1(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1) 
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    return feature_bank


def compu_proto(feature_bank, args):
    # computing prototypes by K-means
    print("Computing prototype for each sample...")
    from sklearn.cluster import KMeans
    features_np = feature_bank.cpu().numpy()
    kmeans = KMeans(n_clusters=args.num_classes, init='random', algorithm='lloyd', random_state = 1)
    kmeans.fit(features_np)
    cluster_labels = kmeans.labels_      
    prototypes = kmeans.cluster_centers_
    sample_prototypes = prototypes[cluster_labels]
    return sample_prototypes


# Training code
def train(train_loader, epoch, model1, optimizer1, args, criterion, prototypes):
    model1.train()
    train_total = 0
    train_correct = 0

    for i, (data, labels, indexes) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        feat1, logits1 = model1(data)
        C = prototypes[indexes.cpu().numpy()]
        C = torch.tensor(C).cuda()

        criterion2 = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(logits1, labels.long())
        lossce = criterion2(logits1, labels.long())
        distances = torch.norm((feat1-C), p=2, dim=1)

        loss_protoreg = args.Lambda1*(distances.mean())+ args.Lambda2*((lossce* distances).mean())  
        
        loss = loss + loss_protoreg

        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        acc = accuracy(logits1, labels, topk=(1,))
        prec1 = float(acc[0])
        train_total += 1
        train_correct += prec1

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, args.train_len // args.batch_size, prec1, loss.item()))


    train_acc1 = float(train_correct) / float(train_total)
    return train_acc1


# Evaluate the model.
def evaluate(test_loader, model1):
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0

    with torch.no_grad():
        for data, labels, _ in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            _, logits1 = model1(data)
            _, pred1 = torch.max(logits1.data, 1)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)
    
    return acc1

def main(args):
    save_dir = args.result_dir + args.dataset + '/' + args.noise_type + '/' + str(args.noise_rate) + '/'

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    txtfile = save_dir + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_dataset, feat_dataset, val_dataset, test_dataset = load_data(args)

    train_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    
    feat_loader = torch.utils.data.DataLoader(dataset= feat_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)


    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    # Define models
    print('Building model...')

    if args.dataset == 'mnist':
        clf1 = LeNet()
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 15], gamma=0.1)
    elif args.dataset == 'fmnist':
        clf1 = resnet.ResNet18(input_channel=1, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'cifar10':
        clf1 = resnet.ResNet18(input_channel=3, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[20,  80], gamma=0.1)
    elif args.dataset == 'cifar100':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=100)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[30, 80], gamma=0.1)

    if torch.cuda.is_available():
        clf1.cuda()

    feature_bank = feature_extractor(args.model_path, clf1, feat_loader)
    prototypes = compu_proto(feature_bank, args)

    with open(txtfile, "a") as myfile:
        myfile.write('Lambda1:'+str(args.Lambda1) + '  '+'Lambda2:'+ str(args.Lambda2)+ '  ' + '  ' + 'seed:'+str(args.seed ) + '\n')
        myfile.write('epoch  train_acc   val_acc   test_acc  \n')

    epoch = 0
    train_acc = 0

    # evaluate models with random weights
    val_acc = evaluate(val_loader, clf1)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc))

    test_acc = evaluate(test_loader, clf1)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(test_dataset), test_acc))

    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc) + ' ' + str(val_acc) + ' ' + str(test_acc) + "\n")

    train_acc_list = []
    val_acc_list = []
    test_acc_list = [] 


    # train
    for epoch in range(0, args.n_epoch):
        scheduler1.step()
        print('Learning rate: ', optimizer1.state_dict()['param_groups'][0]['lr'])

        train_acc = train(train_loader, epoch, clf1, optimizer1, args, nn.CrossEntropyLoss(), prototypes)

        train_acc_list.append(train_acc)

        val_acc = evaluate(val_loader, clf1)
        val_acc_list.append(val_acc)

        test_acc = evaluate(test_loader, clf1)
        test_acc_list.append(test_acc)        

        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% ' % (
                epoch + 1, args.n_epoch, len(test_dataset), test_acc))

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc) + ' ' + str(val_acc) + ' ' + str(test_acc) + "\n")

        id = np.argmax(np.array(val_acc_list))
        test_acc_max = test_acc_list[id]
        print('*********** Best test accuracy : %.2f  *********************' % test_acc_max)

    with open(txtfile, "a") as myfile:
        print('seed:', args.seed)
        myfile.write('*********** Best test accuracy : %.2f  *********************\n' % test_acc_max)

    return test_acc_max

if __name__ == '__main__':
    best_acc = main(args)
