from FedLab.fedlab.utils.dataset.partition import CIFAR10Partitioner
from FedLab.fedlab.contrib.dataset import partitioned_cifar10
from FedLab.datasets.pickle_dataset import PickleDataset
from FedLab.fedlab.utils.functional import partition_report, evaluate
from FedLab.fedlab.utils.serialization import SerializationTool
from FedLab.fedlab.core.standalone import StandalonePipeline
from FedLab.fedlab.contrib.algorithm.basic_server import SyncServerHandler
# configuration
from FedLab.fedlab.models.cnn import AlexNet_CIFAR10
# client
from FedLab.fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer, SGDClientTrainer
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  
import torch
import sys
import torchvision
import pandas as pd
import seaborn as sns
# from pathlib import Path
from munch import Munch
import pickle

sys.path.append('/home/Federated-pretrained/')
# femnist = PickleDataset(dataset_name="femnist", data_root="/home/Federated-pretrained/FedLab/datasets", \
#                         pickle_root="/home/Federated-pretrained/FedLab/datasets/pickle_datasets")
# trainset = femnist.get_dataset_pickle("train")

args = Munch
args.total_client = 10
args.alpha = 0.5
args.seed = 42
args.preprocess = True
args.cuda = True
args.partition = "dirichlet" # 'shards' , 'iid' , "dirichlet"
args.balance = False
args.device = 0

args.download = False
num_parts_of_datasets = 15
num_classes = 10
seed = 2023
hist_color = '#4169E1'
plt.rcParams['figure.facecolor'] = 'white'
trainset = torchvision.datasets.CIFAR10(root="/home/Federated-pretrained/data/CIFAR10/", train=True, download=args.download)


# perform partition
hetero_dir_part = partitioned_cifar10.PartitionedCIFAR10(root='/home/Federated-pretrained/', 
                                path='/home/Federated-pretrained/data/CIFAR10/',
                                dataname='cifar10',
                                num_clients=num_parts_of_datasets,
                                balance=args.balance, 
                                partition=args.partition, 
                                unbalance_sgm=0.3,
                                dir_alpha=0.3,
                                seed=seed,
                                download=args.download,
                                preprocess=args.preprocess,
                                num_shards = 200,
                                transform=torchvision.transforms.Compose(
                                    [torchvision.transforms.ToTensor()])
                                )
# save to pkl file
# torch.save(hetero_dir_part.client_dict, "cifar10_hetero_dir.pkl")
# print(len(hetero_dir_part))

# generate partition report
csv_file = "/home/Federated-pretrained/partition-reports/cifar10_"+ args.partition +"_"+str(args.balance)+"_"+str(args.total_client)+"clients.csv"
partition_report(trainset.targets, hetero_dir_part.data_indices, 
                 class_num=num_classes, 
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file,header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
col_names = [f"class{i}" for i in range(num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

# select first 10 clients for bar plot
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)  
# plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
plt.savefig(f"/home/Federated-pretrained/imgs/cifar10_"+ args.partition +"_"+str(args.balance)+"_"+str(args.total_client)+"clients.png", dpi=400, bbox_inches = 'tight')
plt.cla()
# plot sample number distribution for clients
clt_sample_num_df = hetero_dir_part.client_sample_count
sns.histplot(data=clt_sample_num_df, 
             x="num_samples", 
             edgecolor='none', 
             alpha=0.7, 
             shrink=0.95,
             color=hist_color)
plt.savefig(f"/home/Federated-pretrained/imgs/cifar10_"+ args.partition +"_"+str(args.balance)+"_"+str(args.total_client)+"clients_dist.png", dpi=400, bbox_inches = 'tight')


model = AlexNet_CIFAR10()


# local train configuration
args.epochs = 5
args.batch_size = 256
args.lr = 0.1

trainer = SGDSerialClientTrainer(model, args.total_client, device=args.device, cuda=args.cuda) # serial trainer
# trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(hetero_dir_part)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)


# global configuration
args.com_round = 80 # 150
args.sample_ratio = 1.0

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)
original_param = handler.downlink_package



# FL-training

loss_list_main, loss_list_in, loss_list_pre, loss_list_post = [], [], [], []
acc_list_main, acc_list_in, acc_list_pre, acc_list_post = [], [], [], []

class EvalPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader 
    def main(self):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            loss_list_main.append(loss)
            acc_list_main.append(acc)
            print("loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
        self.handler.round = 0
    def in_trained_FL_main(self,model):
        criterion = torch.nn.CrossEntropyLoss()
        broadcast = original_param
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            # broadcast = self.handler.downlink_package
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)
            broadcast = self.handler.downlink_package # aggreted-models

            # def warm_start_stage():
            # broadcast_1 = [SerializationTool.serialize_model(model)]
            SerializationTool.deserialize_model(model, broadcast[0])
            # broadcast_2 = [SerializationTool.serialize_model(model)]
            model = model.cuda(args.device)
            optimizer = torch.optim.SGD(model.parameters(), args.lr)
            model.train()
            for _ in range(0,1):  # warm-start stage epoches
                for id in range(10,15):
                    train_loader = hetero_dir_part.get_dataloader(id, 256) # batch size
                    for data, target in train_loader:
                        if args.cuda:
                            data = data.cuda(args.device)
                            target = target.cuda(args.device)

                        output = model(data)
                        loss = criterion(output, target)
                        flood = (loss-0.1).abs()+0.1
                        flood.backward()
                        optimizer.zero_grad()
                        # loss.backward()
                        optimizer.step()
                # return model.get_parameter() # model_parameters
            # warm_start_stage()
            loss, acc = evaluate(model, nn.CrossEntropyLoss(), self.test_loader)
            print("in stage:: loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
            loss_list_in.append(loss)
            acc_list_in.append(acc)
            broadcast = [SerializationTool.serialize_model(model)]

            # loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            # print("loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
        self.handler.round = 0
    def pre_trained_FL_main(self,model):
        # def warm_start_stage():
        SerializationTool.deserialize_model(model, original_param[0])
        model = model.cuda(args.device)
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        # model = model.cuda(args.device)
        for _ in range(50):  # warm-start stage epoches
            for id in range(10,15):
                train_loader = hetero_dir_part.get_dataloader(id, 256) # batch size
                for data, target in train_loader:
                    if args.cuda:
                        data = data.cuda(args.device)
                        target = target.cuda(args.device)

                    output = model(data)
                    loss = criterion(output, target)

                    optimizer.zero_grad()
                    flood = (loss-0.1).abs()+0.1
                    flood.backward()
                    # loss.backward()
                    optimizer.step()
            # return model.get_parameter() # model_parameters
        # warm_start_stage()
        loss, acc = evaluate(model, nn.CrossEntropyLoss(), self.test_loader)
        print("warm-start stage:: loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
        broadcast = [SerializationTool.serialize_model(model)]
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            
            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)
            broadcast = self.handler.downlink_package # aggreted-models

            loss, acc = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
            loss_list_pre.append(loss)
            acc_list_pre.append(acc)
        self.handler.round = 0
    def post_trained_FL_main(self,model):
        criterion = torch.nn.CrossEntropyLoss()
        broadcast = self.handler.downlink_package # aggreted-models

        SerializationTool.deserialize_model(model, broadcast[0])
        model = model.cuda(args.device)
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
        model.train()
        for _ in range(50):  # warm-start stage epoches
            for id in range(10,15):
                train_loader = hetero_dir_part.get_dataloader(id, 256) # batch size
                for data, target in train_loader:
                    if args.cuda:
                        data = data.cuda(args.device)
                        target = target.cuda(args.device)

                    output = model(data)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    flood = (loss-0.1).abs()+0.1
                    flood.backward()
                    # loss.backward()
                    optimizer.step()
        loss, acc = evaluate(model, nn.CrossEntropyLoss(), self.test_loader)
        print("post stage:: loss {:.4f}, test accuracy {:.4f}".format(loss, acc))
        loss_list_main.append(loss)
        acc_list_main.append(acc)
        self.handler.round = 0
test_data = torchvision.datasets.CIFAR10(root="/home/Federated-pretrained/data/CIFAR10/",
                                       train=False,
                                       download=args.download,
                                       transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1024)

standalone_eval = EvalPipeline(handler=handler, trainer=trainer, test_loader=test_loader)

standalone_eval.main()
standalone_eval.post_trained_FL_main(model)
standalone_eval.in_trained_FL_main(model)
standalone_eval.pre_trained_FL_main(model)
# standalone_eval.post_trained_FL_main(model)

loss_dic = {'main':loss_list_main, 'in':loss_list_in, 'pre':loss_list_pre, 'post':loss_list_post}
acc_dic = {'main':acc_list_main, 'in':acc_list_in, 'pre':acc_list_pre, 'post':acc_list_post}
with open("/home/Federated-pretrained/result-reports/cifar10_"+ args.partition +"_"+str(args.balance)+"_"+str(args.total_client)+"clients.pkl", "wb") as save_file:
    pickle.dump([loss_dic, acc_dic], save_file)
