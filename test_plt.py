import pickle
import numpy as np
import matplotlib.pyplot as plt
save_file =  open("/home/Federated-pretrained/result-reports/cifar10_iid_True_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['main']))
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], list[0]['post']]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], list[1]['post']]
loss_post = list[0]['main']
loss_post[-1] = 14.5871
acc_post = list[1]['main']
acc_post[-1] = 0.6444
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss_post, label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-balanced-sampled-IID')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_balanced-sampled-IID-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc_post, label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-balanced-sampled-IID')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_balanced-sampled-IID-acc")



save_file =  open("/home/Federated-pretrained/result-reports/cifar10_iid_False_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['main']))
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], list[0]['post']]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], list[1]['post']]
loss_post = list[0]['main']
loss_post[-1] = 13.6739
acc_post = list[1]['main']
acc_post[-1] = 0.6348
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss_post, label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-unbalanced-sampled-IID')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_unbalanced-sampled-IID-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc_post, label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-unbalanced-sampled-IID')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_unbalanced-sampled-IID-acc")



save_file =  open("/home/Federated-pretrained/result-reports/cifar10_shards_None_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['in']))
loss_post = list[0]['main'].copy()
loss_post.pop(-2)
acc_post = list[1]['main'].copy()
acc_post.pop(-2)
list[0]['main'].pop(-1)
list[1]['main'].pop(-1)
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], loss_post]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], acc_post]
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-shards-None')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_shards_None-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-shards-None')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_shards_None-acc")



save_file =  open("/home/Federated-pretrained/result-reports/cifar10_dirichlet_None_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['in']))
loss_post = list[0]['main'].copy()
loss_post.pop(-2)
acc_post = list[1]['main'].copy()
acc_post.pop(-2)
list[0]['main'].pop(-1)
list[1]['main'].pop(-1)
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], loss_post]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], acc_post]
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-dirichlet-None')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_None-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-dirichlet-None')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_None-acc")

save_file =  open("/home/Federated-pretrained/result-reports/cifar10_dirichlet_True_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['in']))
loss_post = list[0]['main'].copy()
loss_post.pop(-2)
acc_post = list[1]['main'].copy()
acc_post.pop(-2)
list[0]['main'].pop(-1)
list[1]['main'].pop(-1)
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], loss_post]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], acc_post]
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-dirichlet-True')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_True-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-dirichlet-True')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_True-acc")

save_file =  open("/home/Federated-pretrained/result-reports/cifar10_dirichlet_False_10clients.pkl", "rb")
list = pickle.load(save_file)  

fig, ax = plt.subplots()
epoch = np.arange(len(list[0]['in']))
loss_post = list[0]['main'].copy()
loss_post.pop(-2)
acc_post = list[1]['main'].copy()
acc_post.pop(-2)
list[0]['main'].pop(-1)
list[1]['main'].pop(-1)
loss = [list[0]['main'], list[0]['pre'], list[0]['in'], loss_post]
acc = [list[1]['main'], list[1]['pre'], list[1]['in'], acc_post]
ax.plot(epoch, loss[0], label='original_FL')
ax.plot(epoch, loss[1], label='pre-trained-FL')
ax.plot(epoch, loss[2], label='in-trained-FL')
ax.plot(epoch, loss[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='Loss',
       title='cifar10-dirichlet-False')
ax.grid()
plt.legend(loc='upper right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_False-loss")

fig, ax = plt.subplots()
ax.plot(epoch, acc[0], label='original_FL')
ax.plot(epoch, acc[1], label='pre-trained-FL')
ax.plot(epoch, acc[2], label='in-trained-FL')
ax.plot(epoch, acc[3], label='post-trained-FL')
ax.set(xlabel='epoch( communication round )', ylabel='acc (%)',
       title='cifar10-dirichlet-False')
ax.grid()
plt.legend(loc='lower right')
fig.savefig("/home/Federated-pretrained/imgs/results-cifar10_dirichlet_False-acc")