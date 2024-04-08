from numpy import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cuda")

MIN = 9999
import time

starttime = datetime.datetime.now()

print("Load data")
drug_se_ass = np.loadtxt("data/mat_drug_se.txt")
drug_drug_sim = np.loadtxt("data/Similarity_Matrix_Drugs.txt")
drug_drug_sim_dis = np.loadtxt("data/drug_drug_sim_dis.txt")
se_se_sim = np.loadtxt("data/se_seSmilirity.csv", delimiter=",")

one_positive = np.argwhere(drug_se_ass == 1)
one_positive_length = len(one_positive)
one_positive = np.array(one_positive)
np.random.shuffle(one_positive)

zero_positive = np.argwhere(drug_se_ass == 0)
zero_positive_length = len(zero_positive)
zero_positive = np.array(zero_positive)
np.random.shuffle(zero_positive)

positive_database = []
negative_database = []
sum_list = []


def load_data(id, BATCH_SIZE):
    x = []
    y = []
    for j in range(id.shape[0]):
        temp_save = []
        x_A = int(id[j][0])
        y_A = int(id[j][1])
        temp_save.append([x_A, y_A])
        label = drug_se_ass[[x_A], [y_A]]
        x.append([temp_save])
        y.append(label)
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return data2_loader


def Preproces_Data(drug_se_ass, test_id):
    copy_drug_se = drug_se_ass/1
    for i in range(test_id.shape[0]):
        x = int(test_id[i][0])
        y = int(test_id[i][1])
        copy_drug_se[x, y] = 0
    return copy_drug_se


def index_dp(A, k):
    index_drug_p = transpose(nonzero(A))
    random.shuffle(index_drug_p)
    data_1 = array_split(index_drug_p, k, 0)

    index_dp_zero = argwhere(A == 0)
    random.shuffle(index_dp_zero)
    data_0 = array_split(index_dp_zero, k, 0)

    return data_0, data_1


class AAM_Module(nn.Module):
    """ Attribute attention module"""
    def __init__(self, in_dim):
        super(AAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class HGCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(HGCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.diag(torch.ones(4900)))
        self.p = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.p.size(1))
        self.p.data.uniform_(-stdv, stdv)
        if self.use_bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g1: torch.Tensor, g2: torch.Tensor, x: torch.Tensor):
        g = g1 @ self.weight @ g2 @ x @ self.p
        if self.use_bias:
            g += self.bias
        return g


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        self.channel_attention1 = CAM_Module(8)
        self.pos_attention1 = AAM_Module(8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        self.q_chem1 = nn.Parameter(torch.Tensor(4900, 1024))
        self.q_dis1 = nn.Parameter(torch.Tensor(4900, 1024))
        self.hgcn_chem1 = HGCN(4900, 1024)
        self.hgcn_dis1 = HGCN(4900, 1024)

        self.q_chem2 = nn.Parameter(torch.Tensor(1024, 256))
        self.q_dis2 = nn.Parameter(torch.Tensor(1024, 256))
        self.hgcn_chem2 = HGCN(1024, 256)
        self.hgcn_dis2 = HGCN(1024, 256)

        self.mlp1 = nn.Sequential(nn.Linear(16*1*612, 256), nn.LeakyReLU(inplace=True), nn.Dropout())
        self.mlp2 = nn.Sequential(nn.Linear(10312, 128), nn.LeakyReLU(inplace=True), nn.Dropout())
        self.mlp3 = nn.Sequential(nn.Linear(256+128, 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mlp1[0].weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_normal_(self.mlp2[0].weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_normal_(self.mlp3[0].weight, nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, chem_x, chem_g1, chem_g2, chem_h, dis_x, dis_g1, dis_g2, dis_h, x_pos, y_pos):
        # cnn
        x = self.conv1(x)
        x1 = self.channel_attention1(x)
        x2 = self.pos_attention1(x)
        x = x1 + x2
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.mlp1(x)

        chem1 = self.relu(self.hgcn_chem1(chem_g1, chem_g2, chem_x) + dis_h.T@dis_x@self.q_dis1)
        dis1 = self.relu(self.hgcn_dis1(dis_g1, dis_g2, dis_x) + chem_h.T@chem_x@self.q_chem1)
        chem2 = self.relu(self.hgcn_chem2(chem_g1, chem_g2, chem1) + dis_h.T@dis1@self.q_dis2)
        dis2 = self.relu(self.hgcn_dis2(dis_g1, dis_g2, dis1) + chem_h.T@chem1@self.q_chem2)
        input1 = torch.cat([chem2, chem_x], dim=1)
        input2 = torch.cat([dis2, dis_x], dim=1)
        input1 = input1 + input2
        y_pos = y_pos + 708
        input1 = torch.cat([input1[x_pos], input1[y_pos]], dim=1)
        input1 = self.mlp2(input1)
        input1 = torch.cat([x, input1], dim=1)

        x = self.mlp3(input1)
        return x


def g_generator(drug_drug, drug_drug_topk, rs, ss, ss_topk):
    r1 = torch.cat([drug_drug, rs], dim=1)
    r2 = torch.cat([rs.T, ss], dim=1)
    x1 = torch.cat([r1, r2], dim=0)

    r1_topk = torch.cat([drug_drug_topk, rs], dim=1)
    r2_topk = torch.cat([rs.T, ss_topk], dim=1)
    h1 = torch.cat([r1_topk, r2_topk], dim=0)

    d = torch.diag((torch.sum(h1, dim=1)) ** -1)
    b = torch.diag((torch.sum(h1, dim=0)) ** -1)
    h2 = h1.T
    g1 = d @ h1
    g2 = b @ h2

    return x1, g1, g2, h1


def max_row(matrix, k):
    max_indices = np.argsort(-matrix, axis=0)[:k, :]
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            if i not in max_indices[:, j]:
                matrix[i][j] = 0
    return matrix


def Split_data(data_1, data_0, fold, k):
    X_train = []
    X_test = []
    for i in range(k):
        if i != fold:
            for j in range(len(data_1[i])):
                X_train.append(data_1[i][j])
            for t in range(len(data_0[i])):
                if t < len(data_1[i]):
                    X_train.append(data_0[i][t])
                else:
                    x = int(data_0[i][t][0])
                    y = int(data_0[i][t][1])
                    X_test.append([x, y])
        else:
            for t1 in range(len(data_1[i])):
                x = int(data_1[i][t1][0])
                y = int(data_1[i][t1][1])
                X_test.append([x, y])
            for t2 in range(len(data_0[i])):
                x = int(data_0[i][t2][0])
                y = int(data_0[i][t2][1])
                X_test.append([x, y])
    random.shuffle(X_train)
    return X_train, X_test

net = Net()
net.to(device)
TPR_ALL = []
FPR_ALL = []
P_ALL = []
epochs = 30
fold = 5
data_0, data_1 = index_dp(drug_se_ass, 5)

for test in range(fold):
    train_data = []
    test_data = []
    train_data, test_data = Split_data(data_1, data_0, test, 5, drug_se_ass)
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    new_drug_se = Preproces_Data(drug_se_ass, test_data)

    drug_drug_sim_sub_topk = max_row(drug_drug_sim, 100)
    drug_drug_sim_func_topk = max_row(drug_drug_sim_dis, 100)
    se_se_sim_topk = max_row(se_se_sim, 500)

    ds = torch.FloatTensor(new_drug_se)
    ddc = torch.FloatTensor(drug_drug_sim)
    ddd = torch.FloatTensor(drug_drug_sim_dis)
    ss = torch.FloatTensor(se_se_sim)
    ddc_topk = torch.FloatTensor(drug_drug_sim_sub_topk)
    ddd_topk = torch.FloatTensor(drug_drug_sim_func_topk)
    ss_topk = torch.FloatTensor(se_se_sim_topk)

    chem_x, chem_g1, chem_g2, chem_h = g_generator(ddc, ddc_topk, ds, ss, ss_topk)
    dis_x, dis_g1, dis_g2, dis_h = g_generator(ddd, ddd_topk, ds, ss, ss_topk)

    train_loader = load_data(train_data, 64)
    test_loader = load_data(test_data, 256)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
    loss_func = nn.CrossEntropyLoss()

    chem_x = chem_x.to(device)
    chem_g1 = chem_g1.to(device)
    chem_g2 = chem_g2.to(device)
    chem_h = chem_h.to(device)
    dis_x = dis_x.to(device)
    dis_g1 = dis_g1.to(device)
    dis_g2 = dis_g2.to(device)
    dis_h = dis_h.to(device)
    # train
    print("Training begin")
    for epoch in range(1, epochs + 1):
        since = time.time()
        train_loss = 0
        train_acc = 0
        for step, (x, train_label) in enumerate(train_loader):
            net.train()
            train_label1 = []
            drug_se_attr = []
            for j in range(x.shape[0]):
                temp_save = []
                x_A = int(x[j][0][0][0])
                y_A = int(x[j][0][0][1])
                row_1 = np.concatenate((drug_drug_sim[x_A], new_drug_se[x_A],), axis=0)
                row_2 = np.concatenate((new_drug_se.T[y_A], se_se_sim[y_A],), axis=0)
                row_3 = np.concatenate((drug_drug_sim_dis[x_A], new_drug_se[x_A]), axis=0)
                temp_save.append(row_1)
                temp_save.append(row_2)
                temp_save.append(row_3)
                temp_save.append(row_2)
                temp_save = np.array(temp_save)
                temp_save = temp_save.reshape(2, 2, 4900)
                drug_se_attr.append(temp_save)
            x_pos = x[:, 0, 0, 0]
            y_pos = x[:, 0, 0, 1]
            x_pos, y_pos = x_pos.long().to(device), y_pos.long().to(device)
            train_label1.extend(train_label)
            train_label1 = []
            train_label1.extend(train_label)
            y = torch.LongTensor(np.array(train_label1).astype(int64))
            y = Variable(y).to(device)
            drug_se_attr = torch.FloatTensor(drug_se_attr)
            drug_se_attr = Variable(drug_se_attr).to(device)
            out = net(drug_se_attr, chem_x, chem_g1, chem_g2, chem_h, dis_x, dis_g1, dis_g2, dis_h, x_pos, y_pos)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == y).sum().item()
            acc = num_correct/x.shape[0]
            train_acc += acc
        print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
    # test
    net.eval()
    test_acc = 0
    num_cor = 0
    o = np.zeros((0, 2))
    z = 0
    for test_x, test_label in test_loader:
        z = z + test_x.shape[0]
        test_label1 = []
        drug_se_attr = []
        # c = []
        for j in range(test_x.shape[0]):
            temp_save = []
            x_A = int(test_x[j][0][0][0])
            y_A = int(test_x[j][0][0][1])
            row_1 = np.concatenate((drug_drug_sim[x_A], new_drug_se[x_A]), axis=0)
            row_2 = np.concatenate((new_drug_se.T[y_A], se_se_sim[y_A]), axis=0)
            row_3 = np.concatenate((drug_drug_sim_dis[x_A], new_drug_se[x_A]), axis=0)
            temp_save.append(row_1)
            temp_save.append(row_2)
            temp_save.append(row_3)
            temp_save.append(row_2)
            temp_save = np.array(temp_save)
            temp_save = temp_save.reshape(2, 2, 4900)
            drug_se_attr.append(temp_save)
        x_pos = test_x[:, 0, 0, 0]
        y_pos = test_x[:, 0, 0, 1]
        x_pos, y_pos = x_pos.long().to(device), y_pos.long().to(device)
        test_label1.extend(test_label)
        y = torch.LongTensor(np.array(test_label1).astype(int))
        y = Variable(y).to(device)
        drug_se_attr = torch.FloatTensor(drug_se_attr)
        drug_se_attr = Variable(drug_se_attr).to(device)
        right_test_out = net(drug_se_attr, chem_x, chem_g1, chem_g2, chem_h, dis_x, dis_g1, dis_g2, dis_h, x_pos, y_pos)
        right_test_out = F.softmax(right_test_out, dim=1)
        _, pred_y = right_test_out.max(1)
        num_correct = (pred_y == y).sum().item()
        num_cor += num_correct
        o = np.vstack((o, right_test_out.detach().cpu().numpy()))
    np.savetxt("data/test_out.txt", o)