# 导包
import torch
import numpy as np
from PINN.toy_1d.DNN import DNN
from pyDOE2 import lhs
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINN_1d_neuman():
    def __init__(self, X_0, X_1, X_t0, u_t0, X_f, layers, lb, ub, nu):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)
        self.x_0 = torch.tensor(X_0[:, 0:1], requires_grad=True).float().to(device)
        self.t_0 = torch.tensor(X_0[:, 1:2], requires_grad=True).float().to(device)
        self.x_1 = torch.tensor(X_1[:, 0:1], requires_grad=True).float().to(device)
        self.t_1 = torch.tensor(X_1[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.x_t0 = torch.tensor(X_t0[:, 0:1], requires_grad=True).float().to(device)
        self.t0 = torch.tensor(X_t0[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u_t0).float().to(device)
        self.layers = layers
        self.nu = nu
        self.dnn = DNN(layers).to(device)

        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=1000, # 最大迭代次数
            max_eval=1000, # 最大评估次数
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0

    def u_x_real(self, x, t):
        return torch.exp(-t) * torch.ones_like(x)*0

    def u_t0_real(self, x, t):
        return torch.cos(x) * torch.ones_like(t)

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        f = u_t - self.nu * u_xx
        return f

    def net_u_x0(self, x, t):
        u = self.net_u(x, t)
        u_x0 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        return u_x0

    def net_u_x1(self, x, t):
        u = self.net_u(x, t)
        u_x1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        return u_x1

    def loss_func(self):
        self.optimizer.zero_grad()
        u_t0pred = self.net_u(self.x_t0, self.t0)
        f_pred = self.net_f(self.x_f, self.t_f)
        g_bc0 = self.u_x_real(self.x_0, self.t_0)
        g_bc1 = -self.u_x_real(self.x_1, self.t_1)
        loss_u = torch.mean((self.u_t0_real(self.x_t0,self.t0) - u_t0pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        loss_bc = (torch.mean((self.net_u_x0(self.x_0, self.t_0) - g_bc0) ** 2) +
                   torch.mean((self.net_u_x1(self.x_1, self.t_1) - g_bc1) ** 2))
        loss = loss_f + loss_bc + loss_u
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print('Iter %d, Loss: %.5e,loss_u:%.5e, Loss_bc: %.5e, Loss_f: %.5e' % (
                self.iter, loss.item(), loss_u.item(), loss_bc.item(), loss_f.item()))
        return loss

    def train(self, n_iter, optimizer_type='adam'):
        # 其中参数n_iter指的是迭代次数 optimizer_type指的是优化器的类型
        self.dnn.train()
        if optimizer_type == 'adam':
            self.optimizer = self.optimizer_adam
        else:
            self.optimizer = self.optimizer_lbfgs
        # n_iter指的是迭代次数
        for _ in range(n_iter):
            self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

x = np.round(np.linspace(0, np.pi, 256),4)# 生成x坐标并保留4位小数
t = np.round(np.linspace(0, 1, 100),4) # 生成t坐标并保留4位小数
x = x.reshape(256,1)
t = t.reshape(100,1)

# 生成在x=0,x=pi的导数
x_0 = np.zeros((100,1)).reshape(100,1)
x_1 = np.pi*np.ones((100,1)).reshape(100,1)
t0 = np.zeros((256,1))
# ux_bsol = np.exp(-t).reshape(100,1)  # x=0
# ux_esol = -np.exp(-t).reshape(100,1) # x=pi
X_0 = np.hstack((x_0, t)) # x=0
X_1 = np.hstack((x_1, t)) # x=pi
X_t0 = np.hstack((x, t0)) # t = 0
'''
create a dataset for train,
the first dimension for the x beginning point;
the second dimension for the x ending point;
the third dimension for the t initial point.
'''
# BoundarySet = np.array([X_0, X_1, X_t0])

# usol = np.zeros((100,256))
usol = np.exp(-t)*np.cos(x).T
u_t0 = usol[0,:]

X,T = np.meshgrid(x,t)

# data set
nu = 1
noise = 0.0

# N_u = 100
N_ux = 50
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20,1]

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = usol.flatten()[:,None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

idx1 = np.random.choice(X_0.shape[0], N_ux, replace=False)
X_0_train = X_0[idx1, :]
idx2 = np.random.choice(X_1.shape[0], N_ux, replace=False)
X_1_train = X_1[idx2, :]
idx3 = np.random.choice(X_t0.shape[0], N_ux, replace=False)
X_t0_train = X_t0[idx3, :]
u_t0_train = u_t0[idx3]

X_f_train = lb + (ub-lb)*lhs(2, N_f)

model = PINN_1d_neuman(X_0_train,X_1_train,X_t0_train,u_t0_train, X_f_train, layers, lb, ub, nu)
model.train(1000)

# evaluate and predict
u_pred, f_pred = model.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

####### Row 0: u(t,x) ##################
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_0_train[:,1],
    X_0_train[:,0],
    'kx', label = 'bundary0 Data\n (%d points)' % (X_0_train.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)
ax.plot(
    X_1_train[:,1],
    X_1_train[:,0],
    'bx', label = 'bundary1 Data\n (%d points)' % (X_1_train.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)
# 画白线
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
# 设置坐标轴
ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.show()

####### Row 1: u(t,x) slices ##################

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(X[25,:],usol[25,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(X[25,:],U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
# 坐标标签的设置
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize = 15)
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([-1,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(X[50,:],usol[50,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(X[50,:],U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([-1,1.1])
ax.set_title('$t = 0.50$', fontsize = 15)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(X[75,:],usol[75,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(X[75,:],U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([-1,1.1])
ax.set_title('$t = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()