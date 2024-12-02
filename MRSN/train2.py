from model import Model,LMF_Model
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from pandas import DataFrame
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

class ResNet18(nn.Module):
    def __init__(self,**kwargs):
        super(ResNet18,self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.base=resnet18
        self.fc1 = nn.Linear(1000,128)
        self.fc2 = nn.Linear(128,4)
        self.relu=nn.ReLU()


    def forward(self,x):
        x1 = self.base(x)
        out=self.fc1(x1)
        out=self.relu(out)
        out1=self.fc2(out)
        out1=self.relu(out1)
    
        return out1,out

class Clinical(Dataset):
    def __init__(self,df,transform=None):
        self.num_cls = len(df)
        self.img_list = []
        for i in range(self.num_cls):
            self.img_list += [[df.iloc[i,0],df.iloc[i,1],df.iloc[i,2],df.iloc[i,3]]]
        # self.img_list=df
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'sfz': self.img_list[idx][0],
                  'clinical': np.array(self.img_list[idx][1]),
                  'img_ft': np.array(self.img_list[idx][2]),
                  'plan': np.array(self.img_list[idx][3])}
        return sample

def train(optimizer, epoch,bs,fold):
    model.train()
    train_loss = 0
    for batch_index, batch_samples in enumerate(train_loader):
        clinical_f,img_f,wb_plan= batch_samples['clinical'].cuda(),batch_samples['img_ft'].cuda(), batch_samples['plan'].cuda().to(torch.float)
        optimizer.zero_grad()
        # output = model(clinical_f.to(torch.float32),img_f.to(torch.float32))
        output = model(torch.cat((clinical_f.to(torch.float32),img_f.to(torch.float32)),dim=1))
        criteria = nn.MSELoss()
        loss = criteria(output,wb_plan)
        # loss += fuse_loss 
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch%20==0:print('Train Epoch: {}   Train set: Average loss: {:.4f}'.format(
        epoch,train_loss/len(train_loader.dataset)))
    if epoch%300==0: torch.save(model.state_dict(),'./checkpoints/MRRN-'+str(fold)+'.pth')
    return train_loss/len(train_loader.dataset)

def test(epoch): 
    model.eval()
    test_loss = 0   
    criteria = nn.MSELoss()
    with torch.no_grad():
        predlist=[]
        scorelist=[]
        targetlist=[]
        feature_list=[]
        for _, batch_samples in enumerate(test_loader):
            sfz,clinical_f,img_f,wb_plan= batch_samples['sfz'],batch_samples['clinical'].cuda(),batch_samples['img_ft'].cuda(), batch_samples['plan'].cuda().to(torch.float)
            # output = model(clinical_f.to(torch.float32),img_f.to(torch.float32))
            output = model(torch.cat((clinical_f.to(torch.float32),img_f.to(torch.float32)),dim=1))
            test_loss += criteria(output,wb_plan)
            targetcpu=wb_plan.cpu().numpy()
            predlist=np.append(predlist,output.cpu().numpy())

            targetlist=np.append(targetlist,targetcpu)
    return targetlist,predlist,test_loss

if __name__ == '__main__':
    model = ResNet18().cuda()
    # model.load_state_dict(torch.load("bone_resnet18.pth", map_location=lambda storage, loc: storage.cuda(6)))
    model.load_state_dict(torch.load("bone_resnet18.pth"))

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])

    clinical_data = pd.read_excel("./clinical_with_plan3.xlsx")
    subplan = pd.read_excel("./tb_sub_plan.xlsx")
    for i in range(len(clinical_data)):
        if clinical_data.iloc[i,6]=='男':
            clinical_data.iloc[i,6]=1
        else:
            clinical_data.iloc[i,6]=0


    scaler = MinMaxScaler()

    clinical_data.年龄 = scaler.fit_transform(np.array(clinical_data.年龄).reshape(-1, 1))
    clinical_data.体重 = scaler.fit_transform(np.array(clinical_data.体重).reshape(-1, 1))
    clinical_data.身高 = scaler.fit_transform(np.array(clinical_data.身高).reshape(-1, 1))
    clinical_data.diagnose = scaler.fit_transform(np.array(clinical_data.diagnose).reshape(-1, 1))
    clinical_data.发病时间 = scaler.fit_transform(np.array(clinical_data.发病时间).reshape(-1, 1))

    xray_dir="./xray_with_plan/"
    clinical_list = []
    im_feat_list=[]
    plan_list = []
    for iii in range(len(clinical_data)):
        user_id = clinical_data.iloc[iii,0]
        body_weight = clinical_data.iloc[iii,4]*100
        sfz = clinical_data.iloc[iii,2]
        user_plan=subplan[subplan["user_id"]==user_id]
        latest_version=np.max(user_plan.version)
        xray=os.path.join(xray_dir,sfz)
        xray=os.path.join(xray,"post/b.JPG")
        if os.path.isfile(xray)!=1:
            xray = xray.replace(".JPG",".jpg")
        if os.path.isfile(xray)!=1:
            xray = xray.replace(".jpg",".png")

        # print(xray)
        xray_img=Image.open(xray).convert('RGB')
        xray_img=transformer(xray_img)
        xray_img=xray_img.unsqueeze(0)
        model.eval()
        _,img_feature=model(xray_img.cuda())
        plan=user_plan[user_plan["version"]==latest_version].load.tolist()

        if len(plan)<=30:
            if len(plan)==0:
                continue
            im_feat_list.append(img_feature.cpu().detach().numpy()[0])
            img_out = np.array(img_feature.cpu().detach().numpy()[0])
            clinical_list.append([clinical_data.iloc[iii,2],clinical_data.iloc[iii,3:].tolist()])
            plan_list.append([plan[0],len(plan)])
    
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])

    transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])

    info_dict = clinical_list
    info_dict=DataFrame(info_dict)
    info_dict.columns=["sfz","clinical"]
    info_dict["img_feature"]=im_feat_list
    info_dict["plan"]=plan_list

    mse_list = []
    
    k = 5

    info_dict = shuffle(info_dict)
    target_weights = []
    predict_weights = []
    target_time = []
    predict_time = []
    for i in range(k):
        train_df=info_dict.drop(np.arange(int(np.floor(len(info_dict)*i/k)),int(np.floor(len(info_dict)*(i+1)/k))))
        test_df=info_dict.iloc[int(np.floor(len(info_dict)*i/k)):int(np.floor(len(info_dict)*(i+1)/k)),:]

        train_df.index=range(len(train_df))
        test_df.index=range(len(test_df))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 16

        train_dataset=Clinical(train_df)
        test_dataset=Clinical(test_df)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)


        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = Model().to(device)
        # model = LMF_Model().to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

        total_epoch = 300
        train_num=range(1,total_epoch+1)
        avg_loss=[]
        loss = 10
        epoch = 0
        for epoch in range(1, total_epoch+1):
            loss=train(optimizer,epoch,batch_size,i)
            avg_loss.append(loss)

        targetlist,predlist,test_loss = test(epoch)
        predlist = np.round(predlist)
        print('target',targetlist)
        print('predict',predlist)
        print("test loss:",test_loss.cpu().numpy()/len(test_loader.dataset))
        mse_list.append(test_loss.cpu().numpy()/len(test_loader.dataset))

        tlen = len(targetlist)
        for n in range(int(tlen/2)):
            target_weights.append(targetlist[2*n])
            predict_weights.append(predlist[2*n])
            target_time.append(targetlist[2*n+1])
            predict_time.append(predlist[2*n+1])

        cv_loss = np.mean(mse_list)
        print("current cv loss:",cv_loss)


    print("CV loss:",np.mean(mse_list))