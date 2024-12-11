# 导包
import numpy as np
from scipy.interpolate import griddata
from pyDOE2 import lhs
from PINN.toy_1d.Pinn import PINN_1d_neuman, PINN_1d_neuman_DGM
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
# RAD
# 定义概率密度函数
def p( residuals, k, c):
    A = np.mean((np.array([res**k for res in residuals])))
    return (residuals**k) / (A + 1e-6) + c

# 定义根据概率密度的抽样
def adaptive_sampling(density, n_samples, x_range):
    norm_density = (density / density.sum()).reshape(-1)
    X_ids = np.random.choice(a=len(x_range), size=n_samples, replace=False, p=norm_density)
    return x_range[X_ids]


# RAR-G
def selectSample(X_train,Xreal,ureal,u,num):
    # num = 100
    # x = np.linspace(0,10,50).reshape(50,1)
    # t = np.linspace(0,1,50).reshape(50,1)
    # Xreal = np.hstack((x,t))
    # ureal = np.arange(0,50).reshape(50,1)
    # u = np.random.rand(50,1)
    err = (u - ureal) ** 2
    topk_indices = np.argsort(err,axis=0)[-num:].reshape(-1)
    X_train = np.vstack((X_train, Xreal[topk_indices,:]))
    return X_train

# 定义残差的热力学图像
def picture(residual):
    residual = residual.reshape(100,256).T
    plt.figure(figsize=(3, 6))
    plt.imshow(residual, extent=[0, np.pi, 0, np.pi], origin='lower', cmap='jet')
    plt.colorbar() # 添加颜色条
    plt.title('PDE residual')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

# 绘制自适应采样之后的散点图
def plot_sample(X):
    # plt.scatter(X_star[:, 0], X_star[:, 1], s=0.2, color='r', label='All samples')
    plt.scatter(X[:, 1], X[:, 0], s=0.2, color='b', label='Selected samples')
    plt.title('Adaptive sampling')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()


# 生成数据集
x = np.round(np.linspace(0, np.pi, 256), 4)
t = np.round(np.linspace(0, 1, 100), 4)
x = x.reshape(256, 1)
t = t.reshape(100, 1)

x_0 = np.zeros((100, 1)).reshape(100, 1)
x_1 = np.pi * np.ones((100, 1)).reshape(100, 1)
t0 = np.zeros((256, 1))

X_0 = np.hstack((x_0, t))
X_1 = np.hstack((x_1, t))
X_t0 = np.hstack((x, t0))

usol = np.exp(-t) * np.sin(x).T
u_t0 = usol[0, :]

X, T = np.meshgrid(x, t)
# 选取训练点个数
nu = 1
N_ux = 50
N_f = 100
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = usol.flatten()[:, None]

lb = X_star.min(0)
ub = X_star.max(0)

idx1 = np.random.choice(X_0.shape[0], N_ux, replace=False)
X_0_train = X_0[idx1, :]
idx2 = np.random.choice(X_1.shape[0], N_ux, replace=False)
X_1_train = X_1[idx2, :]
idx3 = np.random.choice(X_t0.shape[0], N_ux, replace=False)
X_t0_train = X_t0[idx3, :]
u_t0_train = u_t0[idx3]

X_f_train = lb + (ub - lb) * lhs(2, N_f)


# 以下是通过循环里进行RAD采样，与以上不同的是对于初值点同样进行采样
for i in range(1,11):
    # 训练模型
    model2 = PINN_1d_neuman_DGM(X_0_train, X_1_train, X_t0_train, u_t0_train, X_f_train, layers, lb, ub, nu)

    # Train with Adam optimizer
    print("Training with Adam optimizer")
    model2.train(n_iter=1000, optimizer_type='adam')

    # Train with LBFGS optimizer
    # print("Training with LBFGS optimizer")
    # model.train(n_iter=100, optimizer_type='lbfgs')
    u_pred, f_pred = model2.predict(X_star)
    u_t0_pred,f_t0_pred = model2.predict(X_t0)
    # u_t0_pred =model2.net_u(X_t0[0],model2.t0).detach().cpu().numpy() # 将张量数据转化成numpy数组
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    # 计算的配置点和边界点的残差
    residual2 = np.square(u_t0.reshape([len(u_t0),1])-u_t0_pred)
    residual1 = np.square(u_star-u_pred)
    # 计算边界点和配置点（colocations points）概率密度函数
    density1 = p(residual1, 1,1) # 配置点
    density2 = p(residual2, 1,1) # 边界点
    # 归一化概率密度函数
    X_f_train= adaptive_sampling(density1,1000,X_star)
    X_t0_train = adaptive_sampling(density2,50,X_t0)

    if i % 10 == 0:
        picture(residual1)
# 绘制自适应采样之后的散点图
X_converge = np.vstack((X_f_train,X_t0_train))
plot_sample(X_converge)

# 评估与预测
# evaluate and predict
u_pred, f_pred = model2.predict(X_star)

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


