"""plot_single_run.py
==================
Replays a single CSV segment through the TISAC controller and produces
a clean single-run visualization showing:
  - Desired lateral acceleration (the target path)
  - TISAC controller output (what the car actually did)
  - Steer command over time
  - Per-step cost breakdown (lataccel error + jerk)
  - Total cost + component breakdown printed prominently
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── parse optional segment argument ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--segment', default='data/00100.csv',
                    help='Path to CSV segment to replay')
args = parser.parse_args()
SEGMENT = args.segment

# ── imports from your project ────────────────────────────────
from lifter import FourierLifter
from enkf   import KoopmanEnKF
from smpc   import StochasticMPC

# ── tinyphysics-style data structures ────────────────────────
from types import SimpleNamespace

def make_state(row):
    return SimpleNamespace(
        v_ego        = float(row['vEgo']),
        a_ego        = float(row['aEgo']),
        roll_lataccel= float(np.sin(row['roll']) * 9.81),
    )

def make_future_plan(df, idx, horizon=20):
    """Build a future_plan lookahead window the same way tinyphysics does."""
    end = min(idx + horizon, len(df))
    rows = df.iloc[idx:end]
    return SimpleNamespace(
        lataccel     = rows['targetLateralAcceleration'].tolist(),
        v_ego        = rows['vEgo'].tolist(),
        a_ego        = rows['aEgo'].tolist(),
        roll_lataccel= (np.sin(rows['roll'].values) * 9.81).tolist(),
    )

# ── load data ────────────────────────────────────────────────
print(f"Loading {SEGMENT}...")
df = pd.read_csv(SEGMENT).dropna().reset_index(drop=True)
N  = len(df)
target_lataccel  = df['targetLateralAcceleration'].values
t                = np.arange(N) * 0.1   # 10 Hz -> seconds

# ── load K_baseline ──────────────────────────────────────────
K_baseline = np.load('K_baseline.npy')
print(f"K_baseline loaded: {K_baseline.shape}")

# ── initialise controller components ─────────────────────────
lifter = FourierLifter(num_frequencies=3)
enkf   = KoopmanEnKF(K_baseline, n_ensemble=50,
                     process_noise_std=1e-3, measurement_noise_std=1e-1)
smpc   = StochasticMPC()

x_lifted_prev = None
u_prev        = 0.0
d_prev        = None

# ── replay arrays ────────────────────────────────────────────
actual_lataccel = np.zeros(N)
steer_commands  = np.zeros(N)
lataccel_costs  = np.zeros(N)
jerk_costs      = np.zeros(N)

# seed first step with ground truth
actual_lataccel[0] = target_lataccel[0]

print("Replaying controller...")
for k in range(N):
    row         = df.iloc[k]
    tgt         = float(row['targetLateralAcceleration'])
    curr_la     = actual_lataccel[k]
    state       = make_state(row)
    future_plan = make_future_plan(df, k + 1)
    
    # ---- controller.update() logic (inlined so we capture internals) ----
    v_ego     = state.v_ego
    a_ego     = state.a_ego
    road_roll = state.roll_lataccel
    d_curr    = np.array([v_ego, a_ego, road_roll])
    
    x_lifted_curr = lifter.lift_state(curr_la)
    
    if x_lifted_prev is not None:
        enkf.step(x_lifted_k=x_lifted_prev,
                  u_k=u_prev, d_k=d_prev,
                  y_true_next=curr_la)
    K_ensemble = enkf.ensemble
    
    avail = len(future_plan.lataccel)
    future_targets = []
    future_dist    = []
    for i in range(smpc.H):
        idx = min(i, avail - 1) if avail > 0 else 0
        if avail > 0:
            future_targets.append(future_plan.lataccel[idx])
            future_dist.append([future_plan.v_ego[idx],
                                 future_plan.a_ego[idx],
                                 future_plan.roll_lataccel[idx]])
        else:
            future_targets.append(tgt)
            future_dist.append([state.v_ego, state.a_ego, state.roll_lataccel])
            
    optimal_steer = smpc.optimize_steering(
        x_lifted_curr  = x_lifted_curr,
        u_prev         = u_prev,
        target_lataccels = future_targets,
        disturbances   = future_dist,
        K_ensemble     = K_ensemble,
    )
    
    x_lifted_prev = x_lifted_curr
    u_prev        = optimal_steer
    d_prev        = d_curr
    steer_commands[k] = optimal_steer
    
    # propagate lataccel for next step using mean K prediction
    K_mean  = enkf.mean_K() if hasattr(enkf, 'mean_K') else \
              enkf.ensemble.mean(axis=1).reshape(enkf.K_shape)
    omega_k = np.concatenate((x_lifted_curr, [optimal_steer], d_curr))
    if k + 1 < N:
        actual_lataccel[k + 1] = (K_mean @ omega_k)[0]
        
    # cost components (matching tinyphysics scoring)
    lataccel_costs[k] = (curr_la - tgt) ** 2
    
    # Jerk is the rate of change of actual lateral acceleration (dt = 0.1s)
    if k > 0:
        jerk_costs[k] = ((curr_la - actual_lataccel[k-1]) / 0.1) ** 2
    else:
        jerk_costs[k] = 0.0


# ── score (tinyphysics formula) ──────────────────────────────
# Multiply by 100 as per the Comma AI formula
LAT_COST   = float(np.mean(lataccel_costs)) * 100

# Sum and divide by (N-1) since step 0 has no jerk, then multiply by 100
JERK_COST  = float(np.sum(jerk_costs) / (N - 1)) * 100

# Weight lateral tracking by 50
TOTAL_COST = (50 * LAT_COST) + JERK_COST

print(f"\n{'='*45}")
print(f"  Segment : {os.path.basename(SEGMENT)}")
print(f"  Lat Accel Cost : {LAT_COST:.4f}")
print(f"  Jerk Cost      : {JERK_COST:.4f}")
print(f"  TOTAL COST     : {TOTAL_COST:.4f}")
print(f"{'='*45}\n")

# ── plot ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linestyle'    : '--',
})

C_TARGET = '#1A237E'   # deep navy  – desired path
C_ACTUAL = '#E53935'   # red        – controller output
C_STEER  = '#2E7D32'   # green      – steer command
C_LACOST = '#E53935'   # red fill   – lataccel cost
C_JCOST  = '#FB8C00'   # orange     – jerk cost

fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor('#F8F9FA')

gs = gridspec.GridSpec(
    3, 1,
    height_ratios=[2.8, 1.2, 1.2],
    hspace=0.44,
    top=0.88, bottom=0.07,
    left=0.08, right=0.95,
)

# ── big score banner ─────────────────────────────────────────
fig.text(
    0.5, 0.955,
    f'TISAC Controller  —  {os.path.basename(SEGMENT)}',
    ha='center', va='center', fontsize=14, fontweight='bold', color='#1a1a2e',
)
fig.text(
    0.5, 0.925,
    f'Lat Accel Cost: {LAT_COST:.4f}     Jerk Cost: {JERK_COST:.4f}     '
    f'TOTAL COST: {TOTAL_COST:.3f}',
    ha='center', va='center', fontsize=13, fontweight='bold',
    color='#1565C0',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
              edgecolor='#1565C0', linewidth=1.5),
)

# ── PANEL 1: lateral acceleration tracking ───────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('white')

# shade error region between target and actual
ax1.fill_between(t, target_lataccel, actual_lataccel,
                 alpha=0.15, color=C_ACTUAL, label='Tracking error region')
ax1.plot(t, target_lataccel, color=C_TARGET, lw=2.5, alpha=0.9,
         label='Desired lateral accel  (target path)')
ax1.plot(t, actual_lataccel, color=C_ACTUAL, lw=2.0, alpha=0.85,
         ls='--', label='TISAC controller output')

ax1.set_ylabel('Lateral Acceleration  (m/s²)', fontsize=11)
ax1.set_title('Lateral Acceleration Tracking', fontsize=11, pad=5)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.92)
ax1.set_xlim(t[0], t[-1])

# ── PANEL 2: steer command ───────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.set_facecolor('white')

ax2.fill_between(t, steer_commands, alpha=0.20, color=C_STEER)
ax2.plot(t, steer_commands, color=C_STEER, lw=1.6,
         label='Steer command  (SMPC output)')
ax2.axhline(0, color='#999', lw=0.8, ls='-')

ax2.set_ylabel('Steer Command', fontsize=10)
ax2.set_title('SMPC Steering Output', fontsize=10, pad=4)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.92)
ax2.set_xlim(t[0], t[-1])

# ── PANEL 3: per-step cost breakdown ─────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.set_facecolor('white')

ax3.fill_between(t, lataccel_costs, alpha=0.40, color=C_LACOST,
                 label=f'Lat accel cost  (mean={LAT_COST:.4f})')
ax3.fill_between(t, jerk_costs,     alpha=0.40, color=C_JCOST,
                 label=f'Jerk cost       (mean={JERK_COST:.4f})')

ax3.plot(t, lataccel_costs, color=C_LACOST, lw=1.2)
ax3.plot(t, jerk_costs,     color=C_JCOST,  lw=1.2)

ax3.set_xlabel('Time  (s)', fontsize=11)
ax3.set_ylabel('Cost', fontsize=10)
ax3.set_title('Per-Step Cost Breakdown', fontsize=10, pad=4)
ax3.legend(loc='upper right', fontsize=9, framealpha=0.92)
ax3.set_xlim(t[0], t[-1])

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

out = 'tisac_single_run.png'
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved -> {out}")
plt.show()