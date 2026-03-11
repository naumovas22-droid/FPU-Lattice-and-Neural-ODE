# =========================================================
# Neural ODE для Fermi-Pasta-Ulam решетки с вынужденными затухающими осцилляторами
# =========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------- Setup --------------------
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE = torch.float32

# -------------------- Параметры системы --------------------
num_osc = 4  # Количество осцилляторов в решетке
state_dim = 2 * num_osc  # x1,p1,x2,p2,...,xn,pn

# -------------------- Истинные физические параметры (только для генерации данных) --------------------
omega_true = 1.5  # Локальная частота 
gamma_true = 0.3  # Затухание
beta_true = 0.5   # Нелинейный коэффициент FPU-beta
A_true = 1.0      # Амплитуда вынуждения
Omega_true = 0.9  # Частота вынуждения
k_true = 1.0      # Линейный коэффициент связи 

# -------------------- True dynamics --------------------
def true_rhs(t, y):
    # y: [x1, p1, x2, p2, ..., xn, pn]
    xs = y[0::2]
    ps = y[1::2]
    dxs = ps
    dps = np.zeros_like(ps)

    # Граничные условия: x0 = 0, x_{n+1} = 0
    x_ext = np.concatenate(([0.0], xs, [0.0]))

    for i in range(num_osc):
        xi = xs[i]
        pi = ps[i]
        delta_right = x_ext[i+2] - x_ext[i+1]
        delta_left = x_ext[i+1] - x_ext[i]

        # Линейная связь
        linear_coupling = k_true * (x_ext[i+2] + x_ext[i] - 2 * x_ext[i+1])

        # Нелинейная связь FPU-beta
        nonlinear_coupling = beta_true * (delta_right**3 - delta_left**3)

        # Локальный потенциал, затухание, вынуждение (только для первого осциллятора)
        local = -omega_true**2 * xi - gamma_true * pi
        drive = A_true * np.cos(Omega_true * t) if i == 0 else 0.0

        dps[i] = local + linear_coupling + nonlinear_coupling + drive

    dy = np.empty(state_dim, dtype=np.float32)
    dy[0::2] = dxs
    dy[1::2] = dps
    return dy

def integrate_true(y0, t):
    y = np.zeros((len(t), state_dim), dtype=np.float32)
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        k1 = true_rhs(t[i], y[i])
        k2 = true_rhs(t[i]+dt/2, y[i]+dt/2*k1)
        k3 = true_rhs(t[i]+dt/2, y[i]+dt/2*k2)
        k4 = true_rhs(t[i]+dt, y[i]+dt*k3)
        y[i+1] = y[i] + dt*(k1+2*k2+2*k3+k4)/6
    return y

# -------------------- Dataset --------------------
T = 8.0
dt = 0.1
t_np = np.arange(0, T, dt, dtype=np.float32)

N = 120  # Количество траекторий
noise_std = 0.01

data = []
for _ in range(N):
    y0 = np.random.uniform(-0.5, 0.5, size=state_dim).astype(np.float32)
    traj = integrate_true(y0, t_np)
    traj += noise_std * np.random.randn(*traj.shape).astype(np.float32)
    data.append(traj)

data = torch.tensor(np.stack(data), dtype=DTYPE, device=device)
y0_data = data[:, 0]
t = torch.tensor(t_np, dtype=DTYPE, device=device)

# -------------------- Neural ODE --------------------
class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, state_dim)
        )

    def forward(self, t, y):
        inp = torch.cat([y, t], dim=1)
        return self.net(inp)

model = NeuralODE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

# -------------------- RK4 step --------------------
def rk4_step(model, y, t, dt):
    k1 = model(t, y)
    k2 = model(t + dt/2, y + (dt/2) * k1)
    k3 = model(t + dt/2, y + (dt/2) * k2)
    k4 = model(t + dt,   y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# -------------------- Training --------------------
batch_size = 64
epochs = 1000
losses = []
snapshots = {}

for epoch in range(epochs):
    opt.zero_grad()
    idxs = np.random.choice(N, batch_size, replace=False)
    y = y0_data[idxs]

    traj_pred_list = [y]
    for i in range(len(t) - 1):
        ti = t[i].view(1, 1).expand(batch_size, 1)
        y = rk4_step(model, y, ti, dt)
        traj_pred_list.append(y)

    traj_pred = torch.stack(traj_pred_list, dim=1)
    loss = loss_fn(traj_pred, data[idxs])
    loss.backward()
    opt.step()

    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, loss = {loss.item():.6f}")

    if epoch in [50, 100, 200, 400, epochs-1]:
        with torch.no_grad():
            y_vis = y0_data[0:1]
            traj_pred_snap = [y_vis.cpu().numpy()[0]]
            for i in range(len(t)-1):
                ti = t[i].view(1, 1)
                y_vis = rk4_step(model, y_vis, ti, dt)
                traj_pred_snap.append(y_vis.cpu().numpy()[0])
            snapshots[epoch] = np.array(traj_pred_snap)

print("\nГенерируем чистые траектории Neural ODE для восстановления параметров...")
num_eval_trajs = 60
eval_t_list = []
eval_y_list = []

with torch.no_grad():
    for _ in range(num_eval_trajs):
        idx = np.random.randint(0, N)
        y_cur = y0_data[idx:idx+1].clone()
        for i in range(len(t)):
            eval_t_list.append(t[i].view(1, 1))
            eval_y_list.append(y_cur)
            if i < len(t) - 1:
                ti = t[i].view(1, 1)
                y_cur = rk4_step(model, y_cur, ti, dt)

eval_t_all = torch.cat(eval_t_list, dim=0)
eval_y_all = torch.cat(eval_y_list, dim=0)

print("\n" + "="*70)
print("Восстановление физических параметров (LBFGS + 10 стартов + positivity)")
print("="*70)

def parametric_rhs(t, y, omega, gamma, beta, A, Omega):
    # y: batch x state_dim
    # t: batch x 1
    batch_size = y.shape[0]
    xs = y[:, 0::2]
    ps = y[:, 1::2]
    dxs = ps
    dps = torch.zeros(batch_size, num_osc, device=device)

    # Граничные условия: x0=0, x_{n+1}=0 для каждой батч-строки
    x_ext = torch.cat([torch.zeros(batch_size, 1, device=device), xs, torch.zeros(batch_size, 1, device=device)], dim=1)

    for i in range(num_osc):
        xi = xs[:, i]
        pi = ps[:, i]
        delta_right = x_ext[:, i+2] - x_ext[:, i+1]
        delta_left = x_ext[:, i+1] - x_ext[:, i]

        # Линейная связь (k=1)
        linear_coupling = (x_ext[:, i+2] + x_ext[:, i] - 2 * x_ext[:, i+1])

        # Нелинейная связь
        nonlinear_coupling = beta * (delta_right**3 - delta_left**3)

        # Локальный термин и затухание
        local = -omega**2 * xi - gamma * pi

        # Вынуждение только для первого
        drive = A * torch.cos(Omega * t[:, 0]) if i == 0 else torch.zeros_like(t[:, 0])

        dps[:, i] = local + linear_coupling + nonlinear_coupling + drive

    dy = torch.empty(batch_size, state_dim, device=device)
    dy[:, 0::2] = dxs
    dy[:, 1::2] = dps
    return dy

best_loss = float('inf')
best_params = None

for trial in range(10):
    omega_rec = torch.nn.Parameter(torch.tensor(np.random.uniform(1.2, 1.7), dtype=DTYPE, device=device))
    gamma_rec = torch.nn.Parameter(torch.tensor(np.random.uniform(0.1, 0.6), dtype=DTYPE, device=device))
    beta_rec = torch.nn.Parameter(torch.tensor(np.random.uniform(0.3, 0.7), dtype=DTYPE, device=device))
    A_rec     = torch.nn.Parameter(torch.tensor(np.random.uniform(0.6, 1.4), dtype=DTYPE, device=device))
    Omega_rec = torch.nn.Parameter(torch.tensor(np.random.uniform(0.7, 1.1), dtype=DTYPE, device=device))

    optimizer = torch.optim.LBFGS(
        [omega_rec, gamma_rec, beta_rec, A_rec, Omega_rec],
        lr=1.0, max_iter=300, history_size=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        omega_rec.data.clamp_(min=1e-4)
        gamma_rec.data.clamp_(min=1e-4)
        beta_rec.data.clamp_(min=1e-4)
        A_rec.data.clamp_(min=1e-4)
        Omega_rec.data.clamp_(min=1e-4)
        rhs_nn = model(eval_t_all, eval_y_all)
        rhs_param = parametric_rhs(eval_t_all, eval_y_all, omega_rec, gamma_rec, beta_rec, A_rec, Omega_rec)
        loss = nn.MSELoss()(rhs_nn, rhs_param)
        loss.backward()
        return loss

    for _ in range(20):
        optimizer.step(closure)

    final_loss = closure().item()
    print(f"Trial {trial} → loss={final_loss:.8f} | "
          f"ω={omega_rec.item():.4f}  γ={gamma_rec.item():.4f}  β={beta_rec.item():.4f}  "
          f"A={A_rec.item():.4f}  Ω={Omega_rec.item():.4f}")

    if final_loss < best_loss:
        best_loss = final_loss
        best_params = (omega_rec.item(), gamma_rec.item(), beta_rec.item(), A_rec.item(), Omega_rec.item())

omega_f, gamma_f, beta_f, A_f, Omega_f = best_params
print("\n ЛУЧШИЙ РЕЗУЛЬТАТ:")
print(f"   omega  = {omega_f:.6f}   (истина: 1.500000)")
print(f"   gamma  = {gamma_f:.6f}   (истина: 0.300000)")
print(f"   beta   = {beta_f:.6f}   (истина: 0.500000)")
print(f"   A      = {A_f:.6f}   (истина: 1.000000)")
print(f"   Omega  = {Omega_f:.6f}   (истина: 0.900000)")
print("="*70)

# -------------------- Plots --------------------
true_traj = integrate_true(y0_data[0].cpu().numpy(), t_np)

plt.figure(figsize=(6,4))
plt.plot(losses)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Training loss (full trajectory MSE)")
plt.title("Training loss vs epochs (log scale)")
plt.grid()
plt.show()

# Convergence plots for all oscillators
for osc in range(1, num_osc + 1):
    x_idx = 2 * (osc - 1)
    p_idx = 2 * (osc - 1) + 1

    # x
    plt.figure(figsize=(8,5))
    plt.plot(t_np, true_traj[:, x_idx], label=f"True <x{osc}> (noise-free)", linewidth=3)
    for ep, tr in snapshots.items():
        plt.plot(t_np, tr[:, x_idx], "--", label=f"NN Epoch {ep}")
    plt.xlabel("Time t")
    plt.ylabel(f"<x{osc}>")
    plt.title(f"Convergence of Neural ODE trajectories (oscillator {osc})")
    plt.legend()
    plt.grid()
    plt.show()

    # p
    plt.figure(figsize=(8,5))
    plt.plot(t_np, true_traj[:, p_idx], label=f"True <p{osc}> (noise-free)", linewidth=3)
    for ep, tr in snapshots.items():
        plt.plot(t_np, tr[:, p_idx], "--", label=f"NN Epoch {ep}")
    plt.xlabel("Time t")
    plt.ylabel(f"<p{osc}>")
    plt.title(f"Convergence of Neural ODE trajectories (oscillator {osc})")
    plt.legend()
    plt.grid()
    plt.show()

# Robustness plots for all oscillators
final_epoch = epochs - 1
for osc in range(1, num_osc + 1):
    x_idx = 2 * (osc - 1)
    p_idx = 2 * (osc - 1) + 1

    # x
    plt.figure(figsize=(8,5))
    plt.plot(t_np, true_traj[:, x_idx], label="True (noise-free)", linewidth=3)
    plt.plot(t_np, snapshots[final_epoch][:, x_idx], "--", label="Neural ODE (noisy data)")
    plt.xlabel("Time t")
    plt.ylabel(f"<x{osc}>")
    plt.title(f"Robustness to noise (oscillator {osc})")
    plt.legend()
    plt.grid()
    plt.show()

    # p
    plt.figure(figsize=(8,5))
    plt.plot(t_np, true_traj[:, p_idx], label="True (noise-free)", linewidth=3)
    plt.plot(t_np, snapshots[final_epoch][:, p_idx], "--", label="Neural ODE (noisy data)")
    plt.xlabel("Time t")
    plt.ylabel(f"<p{osc}>")
    plt.title(f"Robustness to noise (oscillator {osc})")
    plt.legend()
    plt.grid()
    plt.show()
