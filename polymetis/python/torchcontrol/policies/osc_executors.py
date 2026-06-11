# Copyright (c) 2026.
# Drop-in controllers for the irl_polymetis fork.
#
# Exported from torchcontrol/policies/__init__.py.
#
# Notes on the polymetis RobotModelPinocchio API available in this fork:
#   forward_kinematics(q)            -> (ee_pos, ee_quat[x,y,z,w])
#   compute_jacobian(q)              -> (6, dof) LOCAL_WORLD_ALIGNED geometric Jacobian
#   inverse_dynamics(q, qd, qdd)     -> tau = M(q) qdd + C(q,qd) qd + g(q)   (full RNEA)
#   compute_inertia(q)               -> M(q) (CRBA, one call -- see _mass_matrix)
#   compute_jacobian_time_variation(q, qd) -> (6, dof) dJ  (LOCAL_WORLD_ALIGNED)
#   get_joint_angle_limits()         -> (q_min, q_max)
#   get_joint_velocity_limits()      -> qd_max
# M(q) now comes from a single CRBA call (compute_inertia). The historical 8-RNEA
# inverse_dynamics probe is preserved as _mass_matrix_probe for reference / manual
# fallback against an older .so that predates the compute_inertia binding.

from typing import Dict, List, Optional

import torch

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.utils.tensor_utils import to_tensor, stack_trajectory


class OperationalSpaceTrajectoryExecutor(toco.PolicyModule):
    """Operational Space Controller (Khatib) for an EE pose+twist(+accel) trajectory.

        tau = J^T * Lambda * ( xddot_ff + W .* (Kp e + Kd edot) )      # task
            + N * ( null_kp (q_rest - q)/range - null_kd qdot )        # posture / limit-avoid
            + C(q,qdot) qdot                                           # Coriolis FF (gravity via FCI)

    with  Lambda = (J M^-1 J^T + rho^2 I)^-1   (operational-space inertia, damped)
          Jbar   = M^-1 J^T Lambda             (dynamically consistent inverse)
          N      = I - J^T Jbar^T              (dynamically consistent nullspace projector)
          W      = task_weight                 (6-vector; lower the 3 orientation entries to relax)
    """

    i: int
    dof: int

    def __init__(
        self,
        ee_pose_trajectory: List[T.TransformationObj],
        ee_twist_trajectory: List[torch.Tensor],
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ee_accel_trajectory: Optional[List[torch.Tensor]] = None,
        task_weight=None,               # (6,) weights; e.g. [1,1,1, 0.3,0.3,0.3] to relax orientation
        null_kp: float = 10.0,
        null_kd: float = 2.0,
        q_rest=None,                    # (dof,) preferred posture; default = joint mid-range
        damping: float = 1e-2,          # Levenberg damping rho^2 on Lambda (singularity robustness)
        ignore_gravity: bool = True,    # True: FCI already gravity-compensates
        torque_rate_limit: float = 1000.0,   # Nm/s; matches libfranka rate limits
        use_inertia_shaping: bool = True,     # False -> falls back to plain J^T impedance
    ):
        super().__init__()

        # --- desired trajectory as stacked tensors (indexed once per tick) ---
        self.ee_pos_traj = to_tensor(
            stack_trajectory([p.translation() for p in ee_pose_trajectory])
        )
        self.ee_quat_traj = to_tensor(
            stack_trajectory([p.rotation().as_quat() for p in ee_pose_trajectory])
        )
        self.ee_twist_traj = to_tensor(stack_trajectory(ee_twist_trajectory))
        self.N = self.ee_pos_traj.shape[0]
        if ee_accel_trajectory is not None:
            self.ee_accel_traj = to_tensor(stack_trajectory(ee_accel_trajectory))
        else:
            self.ee_accel_traj = torch.zeros_like(self.ee_twist_traj)

        assert self.ee_pos_traj.shape == torch.Size([self.N, 3])
        assert self.ee_quat_traj.shape == torch.Size([self.N, 4])
        assert self.ee_twist_traj.shape == torch.Size([self.N, 6])

        # --- gains / weights ---
        self.Kp = to_tensor(Kp).reshape(-1)            # (6,)
        self.Kd = to_tensor(Kd).reshape(-1)            # (6,)
        if task_weight is None:
            self.task_weight = torch.ones(6)
        else:
            self.task_weight = to_tensor(task_weight).reshape(-1)

        # --- model & dynamics blocks ---
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            robot_model, ignore_gravity=ignore_gravity
        )

        # --- joint-limit info for nullspace posture (limit avoidance) ---
        q_min, q_max = self.robot_model.get_joint_angle_limits()
        self.q_min = to_tensor(q_min).reshape(-1)
        self.q_max = to_tensor(q_max).reshape(-1)
        self.dof = int(self.q_min.shape[0])
        self._eye = torch.eye(self.dof)
        q_mid = 0.5 * (self.q_min + self.q_max)
        if q_rest is None:
            self.q_rest = q_mid
        else:
            self.q_rest = to_tensor(q_rest).reshape(-1)
        self.q_range = (self.q_max - self.q_min).clamp(min=1e-3)

        self.null_kp = null_kp
        self.null_kd = null_kd
        self.damping = damping
        self.torque_rate_limit = torque_rate_limit
        self.use_inertia_shaping = use_inertia_shaping

        # mutable state -> register as buffer so TorchScript keeps it across ticks
        self.register_buffer("last_torque", torch.zeros(self.dof))
        self.i = 0

    # ---- M(q) via CRBA (single call); see _mass_matrix_probe for the legacy path ----
    def _mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        M = self.robot_model.compute_inertia(q)
        return 0.5 * (M + M.t())                                  # symmetrize numerics

    # ---- legacy 8-RNEA probe of M(q); kept for an .so that predates compute_inertia ----
    # Not called by forward(); swap into _mass_matrix manually if running against an old
    # libtorchscript_pinocchio.so. TorchScript only compiles methods reachable from
    # forward(), so leaving this unused is safe.
    def _mass_matrix_probe(self, q: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(q)
        bias = self.robot_model.inverse_dynamics(q, zero, zero)   # = g(q) (qd=0 -> no Coriolis)
        cols: List[torch.Tensor] = []
        for j in range(self.dof):
            ej = self._eye[:, j]
            tau_j = self.robot_model.inverse_dynamics(q, zero, ej)
            cols.append((tau_j - bias).reshape(self.dof, 1))      # = M[:, j]
        M = torch.cat(cols, dim=1)
        return 0.5 * (M + M.t())                                  # symmetrize numerics

    # ---- orientation error (desired wrt current) as a base-frame rotation vector ----
    def _orientation_error(self, quat_cur: torch.Tensor, quat_des: torch.Tensor) -> torch.Tensor:
        # quaternions are [x, y, z, w]; q_err = q_des (x) conj(q_cur)
        qc = torch.stack([-quat_cur[0], -quat_cur[1], -quat_cur[2], quat_cur[3]])
        d = quat_des
        ew = d[3] * qc[3] - d[0] * qc[0] - d[1] * qc[1] - d[2] * qc[2]
        ex = d[3] * qc[0] + d[0] * qc[3] + d[1] * qc[2] - d[2] * qc[1]
        ey = d[3] * qc[1] - d[0] * qc[2] + d[1] * qc[3] + d[2] * qc[0]
        ez = d[3] * qc[2] + d[0] * qc[1] - d[1] * qc[0] + d[2] * qc[3]
        sign = torch.sign(ew + 1e-12)              # take the shortest geodesic
        v = sign * torch.stack([ex, ey, ez])
        w = sign * ew
        vnorm = torch.linalg.vector_norm(v) + 1e-9
        angle = 2.0 * torch.atan2(vnorm, w)
        return (angle / vnorm) * v

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = state_dict["joint_positions"]
        qd = state_dict["joint_velocities"]

        ee_pos, ee_quat = self.robot_model.forward_kinematics(q)
        J = self.robot_model.compute_jacobian(q)               # (6, dof)
        ee_twist = J @ qd                                      # (6,)

        # desired (indexed by tick; see note on time-indexing for bimanual sync)
        pos_des = self.ee_pos_traj[self.i, :]
        quat_des = self.ee_quat_traj[self.i, :]
        twist_des = self.ee_twist_traj[self.i, :]
        accel_des = self.ee_accel_traj[self.i, :]

        # 6-D task error and twist error
        pos_err = pos_des - ee_pos
        ori_err = self._orientation_error(ee_quat, quat_des)
        x_err = torch.cat([pos_err, ori_err])                 # (6,)
        xd_err = twist_des - ee_twist                         # (6,)

        # commanded task acceleration: feedforward + weighted PD (orientation relaxation via W)
        a_task = accel_des + self.task_weight * (self.Kp * x_err + self.Kd * xd_err)

        if self.use_inertia_shaping:
            M = self._mass_matrix(q)
            Minv = torch.inverse(M)
            Lam = torch.inverse(J @ Minv @ J.t() + self.damping * torch.eye(6))
            F = Lam @ a_task                                  # task wrench
            Jbar = Minv @ J.t() @ Lam                         # (dof, 6)
            N = self._eye - J.t() @ Jbar.t()                  # (dof, dof)
        else:
            F = a_task
            N = self._eye

        tau_task = J.t() @ F

        # nullspace posture -> pulls toward q_rest / mid-range, damps -> joint-limit avoidance
        dq_rest = (self.q_rest - q) / self.q_range
        tau_posture = self.null_kp * dq_rest - self.null_kd * qd
        tau_null = N @ tau_posture

        # Coriolis feedforward (gravity handled by FCI because ignore_gravity=True)
        tau_coriolis = self.invdyn(q, qd, torch.zeros_like(q))

        tau = tau_task + tau_null + tau_coriolis

        # torque-rate limiting at 1 kHz (dt = 1e-3 s)
        dtau = torch.clamp(
            (tau - self.last_torque) / 1e-3, -self.torque_rate_limit, self.torque_rate_limit
        )
        tau = self.last_torque + dtau * 1e-3
        self.last_torque = tau

        self.i += 1
        if self.i == self.N:
            self.set_terminated()
        return {"joint_torques": tau}


class JointTrajectoryComputedTorque(toco.PolicyModule):
    """Computed-torque (inverse-dynamics) tracker for a *joint* trajectory.

        tau = M(q) ( qddot_des + Kq e + Kqd edot ) + C(q,qdot) qdot      (gravity via FCI)

    This is the SOTA choice for KNOWN trajectories: feasibility, sync, mirroring and
    velocity limits are all resolved offline; the runtime law is simple and robust,
    with no runtime IK and no singularity handling needed. It is the strict upgrade of
    the stock JointTrajectoryExecutor, which passes qddot_des = 0 and applies the PD
    as raw torque instead of through M(q).

    Time-indexing: pass ``dt`` (the trajectory sample period) to advance the plan by
    elapsed wall-clock time from the first tick rather than a raw tick counter, so two
    arms stay in lock-step even if a tick is dropped/stretched (osc_franka_notes.md S5).
    Leave ``dt <= 0`` to fall back to the simple per-tick counter.
    """

    i: int
    use_time_index: bool

    def __init__(
        self,
        joint_pos_trajectory: List[torch.Tensor],
        joint_vel_trajectory: List[torch.Tensor],
        joint_acc_trajectory: List[torch.Tensor],
        Kq,
        Kqd,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
        dt: float = -1.0,
    ):
        super().__init__()
        self.q_traj = to_tensor(stack_trajectory(joint_pos_trajectory))
        self.qd_traj = to_tensor(stack_trajectory(joint_vel_trajectory))
        self.qdd_traj = to_tensor(stack_trajectory(joint_acc_trajectory))
        self.N = self.q_traj.shape[0]
        assert self.q_traj.shape == self.qd_traj.shape == self.qdd_traj.shape

        self.Kq = to_tensor(Kq).reshape(-1)
        self.Kqd = to_tensor(Kqd).reshape(-1)
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            robot_model, ignore_gravity=ignore_gravity
        )
        self.i = 0
        self.dt = float(dt)
        self.use_time_index = dt > 0.0
        # t0 < 0 flags "first tick not seen yet"; set on the first forward() call.
        # The server provides state_dict["timestamp"] as an int32 [seconds, nanos]
        # pair, monotonic since robot start (torch_server_ops.cpp).
        self.register_buffer("t0", torch.tensor(-1.0, dtype=torch.float64))

    def _index(self, state_dict: Dict[str, torch.Tensor]) -> int:
        if self.use_time_index and "timestamp" in state_dict:
            ts = state_dict["timestamp"].to(torch.float64)
            t = ts[0].item() + ts[1].item() * 1e-9            # seconds (float)
            if self.t0.item() < 0.0:
                self.t0 = torch.tensor(t, dtype=torch.float64)
            idx = int((t - self.t0.item()) / self.dt)
            if idx < 0:
                idx = 0
            if idx > self.N - 1:
                idx = self.N - 1
            return idx
        return self.i

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = state_dict["joint_positions"]
        qd = state_dict["joint_velocities"]

        idx = self._index(state_dict)
        q_des = self.q_traj[idx, :]
        qd_des = self.qd_traj[idx, :]
        qdd_des = self.qdd_traj[idx, :]

        qdd_cmd = qdd_des + self.Kq * (q_des - q) + self.Kqd * (qd_des - qd)
        tau = self.invdyn(q, qd, qdd_cmd)        # M qdd_cmd + C qd  (+ g if not ignored)

        self.i += 1
        if idx >= self.N - 1:
            self.set_terminated()
        return {"joint_torques": tau}
