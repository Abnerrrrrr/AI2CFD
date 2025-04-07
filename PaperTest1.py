"""
这是训练初值的代码
"""
import numpy as np
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 使用全连接神经网络进行训练(DNN)
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.ReLU
        layer_list = list()
        for i in range(self.depth - 1):
            layer = torch.nn.Linear(layers[i], layers[i + 1])
            # He 初始化
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            layer_list.append(('layer_%d' % i, layer))
            layer_list.append(('bn_%d' % i, torch.nn.BatchNorm1d(layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        last_layer = torch.nn.Linear(layers[-2], layers[-1])
        torch.nn.init.xavier_normal_(last_layer.weight)
        layer_list.append(('layer_%d' % (self.depth - 1), last_layer))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        # 假设输出的最后一个维度是 4，将其拆分为 4 个张量
        out = self.layers(x)
        # return out   ->return one dimentional vector
        N1, N2, N3, N4 = torch.split(out, 1, dim=1)
        return N1, N2, N3, N4

class Multi_hPINN():
    def __init__(self, X_train1, ans_1,X_train2,ans_2,X_1 ,layers):
        # data
        self.x_t0 = torch.tensor(X_train1[:, 0:1], requires_grad=True).float().to(device)
        self.y_t0 = torch.tensor(X_train1[:, 1:2], requires_grad=True).float().to(device)
        self.t0 = torch.tensor(X_train1[:, 2:3], requires_grad=True).float().to(device)
        self.x_1 = torch.tensor(X_1[:, 0:1], requires_grad=True).float().to(device)
        self.y_x1 = torch.tensor(X_1[:, 1:2], requires_grad=True).float().to(device)
        self.t_x1 = torch.tensor(X_1[:, 2:3], requires_grad=True).float().to(device)

        self.x_y1 = torch.tensor(X_train2[:, 0:1], requires_grad=True).float().to(device)
        self.y1 = torch.tensor(X_train2[:, 1:2], requires_grad=True).float().to(device)
        self.t_y1 = torch.tensor(X_train2[:, 2:3], requires_grad=True).float().to(device)
        self.u_y1 = torch.tensor(ans_2[:, 0:1], requires_grad=True).float().to(device)
        self.v_y1 = torch.tensor(ans_2[:, 1:2], requires_grad=True).float().to(device)
        self.T = torch.tensor(ans_1[:, 2:3], requires_grad=True).float().to(device)
        self.rho = torch.tensor(ans_1[:, 3:4], requires_grad=True).float().to(device)
        # parameters
        self.alpha = torch.tensor((5/180)*torch.pi).float().to(device)
        self.U_inf = torch.tensor(1508.55).float().to(device)
        self.T_inf = torch.tensor(226.51).float().to(device)
        self.rho_inf = torch.tensor(1.841e-2).float().to(device)
        self.P_inf = torch.tensor(1.197e3).float().to(device)
        self.C_inf = torch.tensor(301.71).float().to(device)
        self.S = torch.tensor(5.5811e-4).float().to(device)
        self.R = torch.tensor(287).float().to(device)
        self.mu0 = torch.tensor(1.789e-5).float().to(device)
        self.C = torch.tensor(110.4).float().to(device)
        self.T0 = torch.tensor(288.15).float().to(device)
        self.Tve = torch.tensor(3030).float().to(device)
        self.Pr = torch.tensor(0.72).float().to(device)
        self.L = torch.tensor(1.675e-4).float().to(device)
        self.loss = None
        self.layers = layers
        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=100)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer_adam, mode='min', factor=0.5, patience=50, verbose=True
        # )
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=1000,  # 最大迭代次数
            max_eval=1000,  # 最大评估次数
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0
    ##########################----------定义方程----------##################################
    def net_u(self,x,y,t):
        N1,_,_,_ = self.dnn(torch.cat([x, y, t], dim=1))

        # u = (y*cos(alpha))/(x+y) + x*y*N1
        # u = (self.S*y*self.U_inf*torch.cos(self.alpha))/(self.S*y+self.L*x+1.0e-6) + self.L*self.S*x*y*N1
        u = (self.S*y*self.U_inf*torch.cos(self.alpha))/(self.S*y+self.S*self.L*x*(torch.ones_like(y)-y)+1.0e-6) + self.L*self.S**2*x*y*(torch.ones_like(y)-y)*N1
        # u = (self.S*y*self.U_inf*torch.cos(self.alpha))/(self.S*y+self.L*x*(torch.ones_like(y)-y)+1.0e-10) + self.L*self.S**2*x*y*(torch.ones_like(y)-y)*N1
        # u = (y*self.U_inf*torch.cos(self.alpha))/(y+x*(torch.ones_like(y)-y)+1.0e-6) + self.L*self.S**2*x*y*(torch.ones_like(y)-y)*N1
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        return u,u_x,u_y,u_t

    def net_v(self,x,y,t):
        # v = -(y*cos(alpha))/(x+y) + x*y*N1
        _,N2,_,_ = self.dnn(torch.cat([x, y, t], dim=1))
        v = (-self.S*y*self.U_inf*torch.sin(self.alpha))/(self.S*y+self.S*self.L*x*(torch.ones_like(y)-y)+1.0e-6) + self.L*self.S**2*x*y*(torch.ones_like(y)-y)*N2
        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        v_t = torch.autograd.grad(
            v, t,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        return v,v_x,v_y,v_t

    def net_T(self,x,y,t):
        _,_,N3,_ = self.dnn(torch.cat([x, y, t], dim=1))
        # T = T_inf + (S-y)*N3_y-N3
        y0 = torch.zeros_like(y,requires_grad=True)
        _,_,N30,_ = self.dnn(torch.cat([x, y0, t], dim=1))
        N3_y0 = torch.autograd.grad(
            N30,
            y0,
            grad_outputs=torch.ones_like(N30),
            retain_graph=True,
            create_graph=True
        )[0]

        T = self.T_inf + x*(self.S-y)*(N3-N30+(self.S)*N3_y0)
        T_x = torch.autograd.grad(
            T,x,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]
        T_y = torch.autograd.grad(
            T,y,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]
        T_t = torch.autograd.grad(
            T,t,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]
        return T,T_x,T_y,T_t

    def net_rho(self,x,y,t):
        _,_,_,N4 = self.dnn(torch.cat([x, y, t], dim=1))
        # rho = rho_inf + xyN4
        rho = self.rho + x*(self.S-y)*N4
        rho_x = torch.autograd.grad(
            rho,x,
            grad_outputs=torch.ones_like(rho),
            retain_graph=True,
            create_graph=True
        )[0]
        rho_y = torch.autograd.grad(
            rho,y,
            grad_outputs=torch.ones_like(rho),
            retain_graph=True,
            create_graph=True
        )[0]
        rho_t = torch.autograd.grad(
            rho,t,
            grad_outputs=torch.ones_like(rho),
            retain_graph=True,
            create_graph=True
        )[0]
        return rho,rho_x,rho_y,rho_t

    # function
    def function(self,x,y,t):
        # 速度u,v 温度T 密度rho, 压强p,
        u,u_x,u_y,u_t = self.net_u(x,y,t)
        v,v_x,v_y,v_t = self.net_v(x,y,t)
        T,T_x,T_y,T_t = self.net_T(x,y,t)
        rho,rho_x,rho_y,rho_t = self.net_rho(x,y,t)

        p = rho*self.R*T
        p_x = torch.autograd.grad(
            p,x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]
        p_y = torch.autograd.grad(
            p,y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]
        mu_T = self.mu0*((self.T0+self.C)/(T+self.C))*(T/self.T0)**1.5
        cp_T = torch.where(T < 600, 1004.5, (7 / 2) * self.R + self.R * (self.Tve / T) ** 2 * (
                torch.exp(self.Tve / T) / (torch.exp(self.Tve / T) - 1) ** 2))
        k_T = mu_T*cp_T/self.Pr
        part1 = (4/3)* mu_T*u_x - (2/3)*mu_T*v_y
        part1_x = torch.autograd.grad(
            part1,x,
            grad_outputs=torch.ones_like(part1),
            retain_graph=True,
            create_graph=True
        )[0]
        part2 = mu_T*(u_y+v_x)
        part2_x = torch.autograd.grad(
            part2,x,
            grad_outputs=torch.ones_like(part2),
            retain_graph=True,
            create_graph=True
        )[0]
        part2_y = torch.autograd.grad(
            part2,y,
            grad_outputs=torch.ones_like(part2),
            retain_graph=True,
            create_graph=True
        )[0]
        part3 = (4/3)* mu_T*v_y - (2/3)*mu_T*u_x
        part3_y = torch.autograd.grad(
            part3,y,
            grad_outputs=torch.ones_like(part3),
            retain_graph=True,
            create_graph=True
        )[0]
        part4 = k_T*T_x
        part4_x = torch.autograd.grad(
            part4,x,
            grad_outputs=torch.ones_like(part4),
            retain_graph=True,
            create_graph=True
        )[0]
        part5 = k_T*T_y
        part5_y = torch.autograd.grad(
            part5,y,
            grad_outputs=torch.ones_like(part5),
            retain_graph=True,
            create_graph=True
        )[0]
        # equation 1
        f1 = rho_t + u*rho_x+ rho*u_x + v*rho_y + v_x*rho
        # equation 2
        f2 = rho*(u_t+u*u_x+v*u_y) + p_x - part1_x - part2_y
        # equation 3
        f3 = rho*(v_t+u*v_x+v*v_y) + p_y - part3_y - part2_x
        # equation 4
        f4 = (rho*cp_T*(T_t+u*T_x+v*T_y)
              +p*(u_x+v_y)
              -part4_x
              -part5_y
              -(4/3)*mu_T*u_x**2
              -mu_T*u_y**2
              -mu_T*v_x**2
              -(4/3)*mu_T*v_y**2
              +(4/3)*mu_T*u_x*v_y
              -2*mu_T*u_y*v_x
              )
        return f1,f2,f3,f4

    def loss_u(self):
        # f1,_,_,_ = self.function(self.x_t0,self.y_t0,self.t0)
        # loss_f1 = torch.mean(f1**2)
        # upred,u_x,_,_ = self.net_u(self.x_t0,self.y_t0,self.t0)    # 内部
        upred_y1,_,_,_ = self.net_u(self.x_y1,self.y1,self.t_y1)    # 上边界的预测值
        upred_x1,u_x1,_,_ = self.net_u(self.x_1,self.y_x1,self.t_x1)    # 上边界的预测值

        u_xx = torch.autograd.grad(
            u_x1,
            self.x_1,
            grad_outputs=torch.ones_like(u_x1),
            retain_graph=True,
            create_graph = True
        )[0]
        max_u = torch.max((upred_y1 - self.u_y1)**2)
        loss_ubc1 = torch.mean((upred_y1-self.u_y1)**2)
        max_u1 = torch.max(u_xx**2)
        loss_ubc2 = torch.mean(u_xx**2)
        return loss_ubc1/max_u +loss_ubc2/max_u1

    def loss_v(self):
        vpred_y1,_,_,_ = self.net_v(self.x_y1,self.y1,self.t_y1)    # 上边界的预测值
        vpred_x1,_,v_y1,_ = self.net_v(self.x_1,self.y_x1,self.t_x1)    # 上边界的预测值
        v_yy1 = torch.autograd.grad(
            v_y1,
            self.x_1,
            grad_outputs=torch.ones_like(v_y1),
            retain_graph=True,
            create_graph = True
        )[0]
        max_v = torch.max((vpred_y1 - self.v_y1)**2)
        max_v1 = torch.max(v_yy1**2)
        loss_vbc1 = torch.mean((vpred_y1-self.v_y1)**2)
        loss_vbc2 = torch.mean(v_yy1**2)
        return  loss_vbc1/max_v + loss_vbc2/max_v1

    def loss_T(self):
        _, _,f3,_ = self.function(self.x_t0, self.y_t0, self.t0)
        Tpred,_,_,_ = self.net_T(self.x_t0, self.y_t0, self.t0)
        loss_f3 = torch.mean(f3**2)
        loss_Tbc = torch.mean((Tpred-self.T))
        return loss_f3+loss_Tbc

    def loss_rho(self):
        _, _,_,f4 = self.function(self.x_t0, self.y_t0, self.t0)
        rho_pred, _, _, _ = self.net_T(self.x_t0, self.y_t0, self.t0)
        loss_f4 = torch.mean(f4 ** 2)
        loss_Tbc = torch.mean((rho_pred - self.rho))
        return loss_f4 + loss_Tbc

    def loss_function(self):
        self.optimizer.zero_grad()
        loss_u = self.loss_u()
        loss_v = self.loss_v()
        # loss_T = self.loss_T()
        # loss_rho = self.loss_rho()
        loss = loss_u +loss_v # +loss_rho+loss_T
        # loss = loss_u +loss_v+loss_rho+loss_T
        self.loss = loss

        loss.backward()
        self.iter += 1
        # if self.iter % 100 == 0:
        #     print(
        #         'Iter %d, Loss: %.5e, Loss_u: %.5e,Loss_v: %.5e,Loss_rho: %.5e,Loss_T: %.5e' % (
        #             self.iter, loss.item(), loss_u.item(),loss_v.item(),loss_rho.item(),loss_T.item())
        #     )
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e,Loss_v: %.5e' % (
                    self.iter, loss.item(), loss_u.item(),loss_v.item())
            )
            # 打印各层梯度范数
            total_norm = 0
            for p in self.dnn.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient Norm: {total_norm:.4e}")
        # self.scheduler.step(self.loss)  # 动态调整学习率
        return loss

    def train(self, n_iter, optimizer_type='adam'):
        '''训练模型 其中可以更改优化器的类型，以及训练迭代的设置'''
        # 其中参数n_iter指的是迭代次数 optimizer_type指的是优化器的类型
        self.dnn.train()
        if optimizer_type == 'adam':
            self.optimizer = self.optimizer_adam
        else:
            self.optimizer = self.optimizer_lbfgs

        # n_iter指的是迭代次数
        for _ in range(n_iter):
            print(self.loss_function)
            self.optimizer.step(self.loss_function)
            if (self.loss < 1e-6):
                break
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), max_norm=1.0)

    def predict(self,X):
        x = torch.tensor(X[:,0:1], requires_grad=True).float().to(device)
        y = torch.tensor(X[:,1:2], requires_grad=True).float().to(device)
        t = torch.tensor(X[:,2:3], requires_grad=True).float().to(device)
        self.dnn.eval()
        u,u_x,_,_ = self.net_u(x,y,t)
        u_xx = torch.autograd.grad(
            u_x,x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u = u.detach().cpu().numpy()
        u_xx = u_xx.detach().cpu().numpy()

        v,_,v_y,_ = self.net_v(x,y,t)
        v_yy = torch.autograd.grad(
            v_y,y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True
        )[0]
        v = v.detach().cpu().numpy()
        v_yy = v_yy.detach().cpu().numpy()

        return u,u_xx,v,v_yy



# 定义一些参数
alpha = (5/180)*np.pi
U_inf = 1508.55
T_inf = 226.51
rho_inf = 1.841e-2
####-------------------生成数据-----------------####
x = np.linspace(0, 1.675e-4, 100).reshape(-1, 1)
y = np.linspace(0, 5.5811e-4, 100).reshape(-1,1)
x_size = len(x)
y_size = len(y)
t0 =  np.zeros((x_size*y_size, 1))  # 10000*1
X, Y = np.meshgrid(x, y)


# 边界点，初始点数据的生成
# x_0 = np.zeros_like(x)
x_0 = np.zeros_like(x)
x_1 = 1.675e-4 * np.ones_like(x)
y_0 = np.zeros_like(y)
y_1 = 5.5811e-4 * np.ones_like(y)

u_x0 = U_inf*np.cos(alpha)*np.ones_like(x)
u_y1 = U_inf*np.cos(alpha)*np.ones_like(x)
u_y0 = np.zeros_like(x)
u_t0 = U_inf*np.cos(alpha)*np.ones_like(t0)

v_x0 = -U_inf*np.sin(alpha)*np.ones_like(x)
v_y1 = -U_inf*np.sin(alpha)*np.ones_like(x)
v_y0 = np.zeros_like(x)
v_t0 = -U_inf*np.sin(alpha)*np.ones_like(t0)

T_x0 = T_inf*np.ones_like(x)
T_y1 = T_inf*np.ones_like(x)
T_t0 = T_inf*np.ones_like(t0)
# another condition partial T/partial y = 0

rho_x0 = rho_inf*np.ones_like(x)
rho_y1 = rho_inf*np.ones_like(x)
rho_t0 = rho_inf*np.ones_like(t0)
# 生成的数据集
X_0 = np.hstack((x_0, y, np.zeros_like(x)))  # 左边界  # 100*3
X_1 = np.hstack((x_1, y, np.zeros_like(x)))  # 右边界  # 100*3
Y_0= np.hstack((x,y_0,np.zeros_like(x)))     # 下边界  # 100*3
X_y1= np.hstack((x,y_1,np.zeros_like(x)))     # 上边界  # 100*3
X_t0 = np.hstack((X.flatten()[:,None],Y.flatten()[:,None],t0))

ans_X_0 = np.hstack((u_x0,v_x0,T_x0,rho_x0))  # 100*4
ans_y1 = np.hstack((u_y1,v_y1,T_y1,rho_y1))  # 100*4
ans_t0 = np.hstack((u_t0,v_t0,T_t0,rho_t0))  # 10000*4

X_train1 = X_t0
u_train1 = ans_t0
X_train2 = X_y1
u_train2 = ans_y1

x_max = 1.675e-4
y_max = 5.5811e-4

# # 修改数据生成部分
X_train1[:, 0:1] = X_train1[:, 0:1] / x_max  # 归一化x
X_train1[:, 1:2] = X_train1[:, 1:2] / y_max  # 归一化y
X_train2[:, 0:1] = X_train2[:, 0:1] / x_max
X_train2[:, 1:2] = X_train2[:, 1:2] / y_max
# 定义网络层数
layers = [3, 100,90,80,60,50,40,20, 4]
model = Multi_hPINN(X_train1, u_train1, X_train2, u_train2,X_1, layers)
model.train(1000)
# model.generate_graph()

#--------------------绘图--------------------#
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

# x = np.linspace(0, 1.675e-4, 100)
# y = np.linspace(0, 5.5811e-4, 100)
# u_t0_pred,_ = model.predict(X_train1)
# fig = plt.figure()
# ax = plt.axes()
# def create_image(ax, x, y, u_pred):
#     h = ax.imshow(u_pred.T, interpolation='nearest', cmap='rainbow',
#                   extent=[x.min(), x.max(), y.min(), y.max()],
#                   origin='lower', aspect='auto')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.10)
#     cbar = plt.colorbar(h, cax=cax)
#     cbar.ax.tick_params(labelsize=12)
#     ax.set_xlabel('$x$', size=12)
#     ax.set_ylabel('$y$', size=12)
#     ax.set_title(f'$u(x,y,t)$', fontsize=12)
#     ax.tick_params(labelsize=12)
# create_image(ax,x,y,u_t0_pred)

import matplotlib.ticker as mticker

x = np.linspace(0, 1.675e-4, 100)
y = np.linspace(0, 5.5811e-4, 100)

# 将坐标转换为以 1e-4 为单位的数值（避免科学计数法）
x_plot = x / 1e-4  # 数值范围变为 0 ~ 1.675
y_plot = y / 1e-4  # 数值范围变为 0 ~ 5.5811

u_t0_pred,u_xx_pred,v_t0_pred,v_yy_pred = model.predict(X_train1)

# 绘制线图
def plot_curve(x, y,y_p,title, xlim=None, ylim=None):
    """
    绘制 x 和 y 的曲线图。

    参数:
        x (list or numpy.ndarray): x 坐标数据
        y (list or numpy.ndarray): y 坐标数据
        xlim (tuple, optional): x 轴的范围，格式为 (xmin, xmax)。如果为 None，则自动调整。
        ylim (tuple, optional): y 轴的范围，格式为 (ymin, ymax)。如果为 None，则自动调整。
    """
    # 创建图形和轴
    plt.figure()
    # 绘制曲线
    plt.plot(x, y, label="True")
    plt.plot(x, y_p, label="Predict",linestyle='--')
    # 设置标题和坐标轴标签
    plt.title(f'{title}')
    plt.xlabel("x")
    plt.ylabel("y")
    # 如果指定了坐标范围，则设置坐标轴范围
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()

# 绘制二维图
def create_image2D(ax, x, y, u_pred,title):
    # 确保预测结果形状匹配网格 (100x100)
    u_pred_2d = u_pred.reshape(len(x), len(y))  # 根据数据顺序调整转置

    # 绘制热力图，extent参数使用转换后的坐标范围
    h = ax.imshow(u_pred_2d,
                  interpolation='nearest',
                  cmap='rainbow',
                  extent=[x.min(), x.max(), y.min(), y.max()],  # 使用转换后的坐标范围
                  origin='lower',
                  aspect='auto')

    # 坐标轴标签显示 1e-4 单位
    ax.set_xlabel('$x$ (×1e−4 m)', fontsize=12)
    ax.set_ylabel('$y$ (×1e−4 m)', fontsize=12)

    # 自定义刻度格式（例如 0.5 代表 0.5e-4）
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # 颜色条设置
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('$u$ Value', fontsize=12)

    # 标题和刻度调整
    ax.set_title(f'{title}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
#-------------------------绘制二维图-------------------------#
# 绘制预测的u的值
fig1 = plt.figure()
ax1 = plt.axes()
create_image2D(ax1, x_plot, y_plot, u_t0_pred,'u_2D vs up_2D graph')
plt.show()
# 绘制预测的u_xx的值
fig2 = plt.figure()
ax2 = plt.axes()
create_image2D(ax2, x_plot, y_plot, u_xx_pred,'u_xx2D vs up_xx2D graph')
plt.tight_layout()
plt.show()

#------------------v的二维图------------------#
fig3 = plt.figure()
ax3 = plt.axes()
create_image2D(ax3, x_plot, y_plot, v_t0_pred,'v_2D vs vp_2D graph')
plt.show()
# 绘制预测的u_xx的值
fig4 = plt.figure()
ax4 = plt.axes()
create_image2D(ax4, x_plot, y_plot, v_yy_pred,'v_yy2D vs vp_yy2D graph')
plt.tight_layout()
plt.show()
# 绘制纯NN()
# fig3 = plt.figure()
# ax3 = plt.axes()
# create_image(ax3, x_plot, y_plot, NN1)
# plt.tight_layout()
# plt.show()


#-------------------------绘制线图-------------------------#
u_y1_pred,u_y1_xxp,_,_ = model.predict(X_train2)
plot_curve(y,u_y1,u_y1_pred,'ureal vs upred curve')

_,_,v_y1_pred,v_y1_yyp = model.predict(X_train2)
plot_curve(y,v_y1,v_y1_pred,' vreal vs vpred curve')
