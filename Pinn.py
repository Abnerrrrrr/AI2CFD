# 导包
import torch
import numpy as np
from PINN.toy_1d.DNN import DNN

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 定义一个类用来求解一维的第一类边界条件问题
class PINN_1d_Dirichlet():
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.layers = layers
        self.nu = nu

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
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
    # ureal 当方程发生更改的时候需要更改这个函数
    def u_real(self, x, t):
        return torch.sin(x) * torch.sinh(t)
    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        # 构造的f，在更换求解方程的时候需要改动
        f = u_xx + u_tt
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u_real(self.x_u,self.t_u) - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
                self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def train(self,n_iter,optimizer_type='adam'):
        self.dnn.train()

        # Backward and optimize
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

# 定义一个类用来求解一维的第二类边界条件问题
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

# 现在同样是第二类边界条件，但使用的是DGM的方法,这里的DGM方法是指的是使用了自适应采样的DGM方法
class PINN_1d_neuman_DGM():
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
        return torch.exp(-t) * torch.ones_like(x)

    def u_t0_real(self, x, t):
        return torch.sin(x) * torch.ones_like(t)

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
    # 现在需要更改其中的loss,将残差平方和的形式更改为MCMC积分的形式
    '''
    func: 传入的 lambda 函数，表示需要计算积分的函数。
    x 和 t: 输入变量。
    num_samples: 采样点的数量。
    该方法生成在定义域内的随机样本点，评估函数值，并计算这些值的均值来近似积分。
    '''
    def monte_carlo_integration(self, func,x,t ,num_samples=1000,s=1):
        # Generate random samples within the domain
        x_samples = torch.rand(num_samples, 1).to(device) * (torch.max(x) - torch.min(x)) + torch.min(x)
        # x_samples = x_samples.requires_grad_(True)
        t_samples = torch.rand(num_samples, 1).to(device) * (torch.max(t) - torch.min(t)) + torch.min(t)
        # t_samples = t_samples.requires_grad_(True)
        # Evaluate the function at the sampled points
        func_values = func(x_samples, t_samples)

        # Compute the Monte Carlo integral as the mean value of the function evaluations
        integral = torch.mean(func_values)*s
        return integral

    def loss_func(self):
        self.optimizer.zero_grad()
        u_t0pred = self.net_u(self.x_t0, self.t0)
        f_pred = self.net_f(self.x_f, self.t_f)
        g_bc0 = self.u_x_real(self.x_0, self.t_0)
        g_bc1 = -self.u_x_real(self.x_1, self.t_1)

        # loss_u = (self.monte_carlo_integration(lambda x, t: (self.u_t0_real(self.x_t0,self.t0) - u_t0pred) ** 2,self.x_t0,self.t0,num_samples=1000,s=np.pi))
        loss_u = 0

        # lambda x, t 是一个匿名函数（lambda 函数），用于定义一个没有名称的简短函数。具体来说，这个 lambda 函数接受两个参数 x 和 t，
        # 并返回 self.net_f(x, t) ** 2 的值。在这种情况下，它用于传递给 monte_carlo_integration 方法以计算残差平方和的蒙特卡罗积分。
        # 这个 lambda 函数的作用是：接受 x 和 t 两个输入计算 self.net_f(x, t) ** 2 的值在 monte_carlo_integration 方法中，
        # func(x_samples, t_samples) 会调用这个 lambda 函数，计算这些样本点上的 self.net_f(x, t) ** 2 的值。


        loss_f = self.monte_carlo_integration(lambda x, t: self.net_f(x, t) ** 2, self.x_f, self.t_f,num_samples=100000,s=np.pi)
        # loss_bc = (self.monte_carlo_integration(lambda x, t: (self.net_u_x0(x, t) - g_bc0) ** 2, self.x_0, self.t_0,num_samples=1000,s=1) +
        #           self.monte_carlo_integration(lambda x, t: (self.net_u_x1(x, t) - g_bc1) ** 2, self.x_1, self.t_1,num_samples=1000,s=1))
        loss_bc = 0
        loss = loss_f + loss_bc + loss_u
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print('Iter %d, Loss: %.5e, loss_u: %.5e, Loss_bc: %.5e, Loss_f: %.5e' % (
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