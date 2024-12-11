# 现在解决下面这个方程
# u_xx+u_yy=0       u(0,y)=0,u(pi,y)=0,u(x,0)=0,u(x,1)=sin(x)sinh(1)
# 真实解：u(x,y)=sin(x)sinh(y)a

# 导包
import numpy as np
from matplotlib import gridspec
from pyDOE2 import lhs
from PINN.toy_1d.Pinn import PINN_1d_Dirichlet
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 生成数据集
x = np.round(np.linspace(0, np.pi, 256), 4)
y = np.round(np.linspace(0, 1, 100), 4)
x = x.reshape(1,256)
y = y.reshape(1,100)
usol = np.zeros((100,256))
usol = np.sinh(y).T*np.sin(x)

X,Y = np.meshgrid(x,y)

# data set
nu = 1
noise = 0.0

N_u = 100
N_f = 20000
layers = [2, 20, 20, 20, 20, 20, 20, 20,20, 1]

X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
u_star = usol.flatten()[:,None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

xx_0 = X[:,0:1]
yy_0 = Y[:,0:1]
X_0 = np.hstack((xx_0, yy_0))
ux_0 = usol[:,0:1]

xx_1 = X[:,-1:]
yx_1 = Y[:,-1:]
X_1 = np.hstack((xx_1, yx_1))
ux_1 = usol[:,-1:]

xy_0 = X[0:1,:].T
yy_0 = Y[0:1,:].T
Y_0 = np.hstack((xy_0, yy_0))
uy_0 = usol[0:1,:].T

xy_1 = X[-1:,:].T
yy_1 = Y[-1:,:].T
Y_1 = np.hstack((xy_1, yy_1))
uy_1 = usol[-1:,:].T

X_u_train = np.vstack([X_0, X_1, Y_0, Y_1])
u_train = np.vstack([ux_0, ux_1, uy_0, uy_1])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

X_f_train = lb + (ub-lb)*lhs(2, N_f)

# 训练
model = PINN_1d_Dirichlet(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
model.train(2000)

# 评估与预测
u_pred, f_pred = model.predict(X_star)
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
Error = np.abs(usol - U_pred)

# 可视化图1
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[y.min(), y.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:,1],
    X_u_train[:,0],
    'kx', label = 'Data (%d points)' % (u_train.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=1.0
)
# 画白线
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(y[0][25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(y[0][50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(y[0][75]*np.ones((2,1)), line, 'w-', linewidth = 1)
# 设置坐标轴
ax.set_xlabel('$y$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(x,y)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

plt.show()

# 可视化图2

####### Row 1: u(t,x) slices ##################

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x[0,:],usol[25,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x[0,:],U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
# 坐标标签的设置
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$y = 0.25$', fontsize = 15)
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([0,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x[0,:],usol[50,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x[0,:],U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([0,1.1])
ax.set_title('$y = 0.50$', fontsize = 15)
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
ax.plot(x[0,:],usol[75,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x[0,:],U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(y,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,np.pi])
ax.set_ylim([0,1.1])
ax.set_title('$y = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()

####################################

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(y[0,:],usol[:,64], 'b-', linewidth = 2, label = 'Exact')
ax.plot(y[0,:],U_pred[:,64], 'r--', linewidth = 2, label = 'Prediction')
# 坐标标签的设置
ax.set_xlabel('$y$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$y = 0.25$', fontsize = 15)
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,1])
ax.set_ylim([0,3])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(y[0,:],usol[:,128], 'b-', linewidth = 2, label = 'Exact')
ax.plot(y[0,:],U_pred[:,128], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,1])
ax.set_ylim([0,3])
ax.set_title('$y = 0.50$', fontsize = 15)
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
ax.plot(y[0,:],usol[:,192], 'b-', linewidth = 2, label = 'Exact')
ax.plot(y[0,:],U_pred[:,192], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$y$')
ax.set_ylabel('$u(y,x)$')
ax.axis('square')
# 如果数据发生更改取值范围同样发生更改
ax.set_xlim([0,1])
ax.set_ylim([0,3])
ax.set_title('$y = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()