import jittor as jt # JIT深度学习框架
from jittor import nn #引入神经网络模块
from jittor import Module as Model  
from PIL import Image # 图像处理
import jclip as clip # 预训练模型是JCLIP
import os # 文件和目录操作
from tqdm import tqdm # 显示进度条
import argparse # 命令行解析
import random # 随机库
import numpy as np # 数学库
from jittor.dataset import Dataset # 引入数据库处理模块
from jittor.dataset import DataLoader # 数据库处理模块的子类 
jt.flags.use_cuda = 1 # 使用GPU
MAX_PER_CLASS = 4 # 少样本分类任务，每个类别只挑选4张图像作为训练的数据集
MAX_TYPE = 374 # 类别数
ratio = 0.9
trainproportion = 0.7

###数据加载部分
img_dir = 'Dataset/'
train_labels = open('Dataset/train.txt').read().splitlines()
train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [int(l.split(' ')[1]) for l in train_labels]

# shuffle
for ind in range(0,3):
    for i in range(len(train_imgs)):
        rd = random.randint(0, len(train_imgs) - 1)
        train_imgs[i], train_imgs[rd] = train_imgs[rd], train_imgs[i] # 随机打乱数据集
        train_labels[i], train_labels[rd] = train_labels[rd], train_labels[i] # 随机打乱标签

cnt = {}
new_train_imgs = []
new_train_labels = []
# 每个类挑四张图，根据train_labels中的label来挑选
for i in range(len(train_imgs)):
    label = train_labels[i]
    if label not in cnt:
        cnt[label] = 0
    if cnt[label] < MAX_PER_CLASS:
        new_train_imgs.append(train_imgs[i])
        new_train_labels.append(train_labels[i])
        cnt[label] += 1

train_features = []
train_labels = new_train_labels
# 加载JCLIP模型和预处理函数
model, preprocess = clip.load("ViT-B-32.pkl")
# calculate image features of training data
print(111)
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = Image.open(os.path.join(img_dir, img))
        img = preprocess(img).unsqueeze(0)
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        train_features.append(image_features)

# 划分训练集和测试集
testproportion = 1 - trainproportion

train_size = int(trainproportion * len(train_features))
test_size = len(train_features) - train_size

test_features = train_features[train_size:]
test_labels = train_labels[train_size:]

train_features = train_features[:train_size]
train_labels = train_labels[:train_size]


train_features = jt.cat(train_features).numpy()
train_labels = jt.int32(train_labels).numpy()

test_features = jt.cat(test_features).numpy()
test_labels = jt.int32(test_labels).numpy()

# 构造训练数据集
class TrainDataset(Dataset):
    def __init__(self, train_features, train_labels):
        super().__init__()
        self.train_features = train_features
        self.train_labels = train_labels

    def __getitem__(self, index):
        return self.train_features[index], self.train_labels[index]

    def __len__(self):
        return len(self.train_features)
    
MytrainDataset = TrainDataset(train_features, train_labels).set_attrs(batch_size=64)

# 构造测试数据集  
class TestDataset(Dataset):
    def __init__(self, test_features, test_labels):
        super().__init__()
        self.test_features = test_features
        self.test_labels = test_labels

    def __getitem__(self, index):
        return self.test_features[index], self.test_labels[index]

    def __len__(self):
        return len(self.test_features)
MytestDataset = TestDataset(test_features, test_labels).set_attrs(batch_size=64)
    

# 读取类别文件
classes = open('Dataset/classes.txt').read().splitlines()
# 将每一类对应的样本标签进行prompt engineering
new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
        c = 'a photo of ' + c + ', a kind of an animal'
    if c.startswith('Thu-dog'):
        c = c[8:]
        c = 'a photo of ' + c + ', a category of a dog'
    if c.startswith('Caltech-101'):
        c = c[12:]
        c = 'a photo of ' + c + ', a kind of an object'
    if c.startswith('Food-101'):
        c = c[9:]
        c = 'a photo of ' + c + ', a type of food'
    new_classes.append(c)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
print('text features matrix has been processed')

# 引入额外的线性全连接层(图像的适配器)
class Adapter(Model):

    def __init__(self,c_in=512,reduction=4):
        super(Adapter,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in,c_in//reduction,bias = False),
            nn.ReLU(),
            nn.Linear(c_in//reduction,c_in,bias = False),
            nn.ReLU()
        )

    def execute(self, x):
        x = nn.dropout(x,0.1)
        x = self.fc(x)
        return x

# 引入新的残差适配网络结构 返回值为一个适配后的图像特征
class CustomJCLIP(Model):

    def __init__(self):
        super().__init__()
        # self.image_encoder = model.encode_image
        self.text_encoder = model.encode_text
        self.dtype = model.dtype
        self.adapter = Adapter() 
    
    def execute(self, image_feature,text_features):
        # image_features = self.image_encoder(image)
        x = self.adapter(image_feature)     
        global ratio # 设置学习比例，0.2的学习后的向量，0.8的原始向量
        image_feature = ratio * x + (1 - ratio) * image_feature
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        predict_vector = (100.0 * image_feature @ text_features.transpose(0, 1)).softmax(dim=-1)
        _ , top_label_predicted = predict_vector.topk(1)
        predict_vector = jt.float32(predict_vector)
        # probability, top_label = text_probs[0].topk(1)
        return predict_vector, top_label_predicted #输出这个预测的概率值分布向量
    
# 实例化模型并进行训练
jclip_adapter = CustomJCLIP()
loss_fn = nn.CrossEntropyLoss()
epoches = 20
batch_size = 64

# 设置优化器
learning_rate =  0.0001
momentum =  0.9
weight_decay =  0.001
optimizer = jt.optim.Adam(jclip_adapter.parameters(), learning_rate, weight_decay)

train_loss = list()
train_acc = list()

test_loss = list()
test_acc = list()


def train(jclip_adapter, dataloader,loss_fn, optimizer,epoch):
    jclip_adapter.train()
    # 保存每一个epoch的loss
    train_losses = list()
    # 记录每一个epoch的准确率
    accuracy = list()
    for batch_idx, (feature, label) in enumerate(dataloader):
    #for i in range(len(dataloader)):
        image_feature = feature
        label = label
        predict_vector = jclip_adapter(image_feature, text_features)[0]
        top_labels = jclip_adapter(image_feature, text_features)[1]
        for i in range(len(label)):
            if top_labels[i] == label[i]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        accurate = sum(accuracy) / len(accuracy)
        loss = loss_fn(predict_vector, label)
        optimizer.step(loss)
        train_losses.append(loss)
        accuracy.append(accurate)
        if batch_idx == 1:
            print('in the {} training epoch, loss is {}, accuracy is {}, '.format(epoch, loss, accurate))
    return train_losses, accuracy

def test(jclip_adapter, dataloader, loss_fn):
    jclip_adapter.eval()
    test_loss = 0
    correct = 0
    # with jt.no_grad():
    for feature, label in dataloader:
        pred_fea,pred = jclip_adapter(feature, text_features)
        test_loss += loss_fn(pred_fea, label)
        if pred == label:
            correct += 1
    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
    return test_loss, correct / len(dataloader.dataset)


for epoch in range(epoches):
    loss,ac = train(jclip_adapter, DataLoader(MytrainDataset, batch_size=batch_size, shuffle=True), loss_fn, optimizer,epoch)
    train_loss.extend(loss)
    train_acc.extend(ac)
    loss, ac = test(jclip_adapter, DataLoader(MytestDataset, batch_size=1, shuffle=False), loss_fn)
    test_loss.append(loss)
    test_acc.append(ac)


print('accuracy in the training set is {}, accuracy in the test set is {}'.format(train_acc[-1],test_acc[-1]))
print('loss in the training set is {}, loss in the test set is {}'.format(train_loss[-1],test_loss[-1]))

# 保存模型
jt.save(jclip_adapter.state_dict(), 'jclip_adapter.pkl')
print('model has been saved')

# 解析命令行参数，默认--split数据集分割指令为A，即使用TestSetA作为测试数据集
parser = argparse.ArgumentParser() 
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()
split = 'TestSet' + args.split

imgs_dir = 'Dataset/' + split
imgs = os.listdir(imgs_dir)

save_file = open('result.txt', 'w')


# 利用plt画图
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
plt.savefig('train_loss.png')
plt.close()

plt.plot(test_loss)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('test_loss.png')
plt.close()

plt.plot(test_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')
plt.close()
# 推理过程
preds = []
with jt.no_grad():
    for img in tqdm(imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        # 利用JCLIP模型提取图像特征
        image_features = jclip_adapter.adapter((model.encode_image(image))) * ratio + model.encode_image(image) * (1 - ratio)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
        # top5 predictions
        _, top_labels = text_probs[0].topk(5)
        preds.append(top_labels)
        # save top5 predictions to file
        save_file.write(img + ' ' +
                        ' '.join([str(p.item()) for p in top_labels]) + '\n')
        
save_file_data  = open('save_file_data.txt', 'w')
save_file_data.write('train_loss: ' + str(train_loss) + '\n')
save_file_data.write('train_acc: ' + str(train_acc) + '\n')
save_file_data.write('test_loss: ' + str(test_loss) + '\n')
save_file_data.write('test_acc: ' + str(test_acc) + '\n')
