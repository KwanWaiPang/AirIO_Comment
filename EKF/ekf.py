import torch
import pypose as pp
from pypose import bmv
from torch import nn
from torch.linalg import pinv



class IMUEKF(nn.Module):
    r'''
    Performs Batched Extended Kalman Filter (EKF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.NLS`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``

    A non-linear system can be described as

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{w}_k,
            \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})  \\
            \mathbf{y}_{k} &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{v}_k,
            \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
        \end{aligned}

    It will be linearized automatically:

    .. math::
        \begin{align*}
            \mathbf{z}_{k+1} = \mathbf{A}_{k}\mathbf{x}_{k} + \mathbf{B}_{k}\mathbf{u}_{k}
                             + \mathbf{c}_{k}^1 + \mathbf{w}_k\\
            \mathbf{y}_{k} = \mathbf{C}_{k}\mathbf{x}_{k} + \mathbf{D}_{k}\mathbf{u}_{k}
                           + \mathbf{c}_{k}^2 + \mathbf{v}_k\\
        \end{align*}

    EKF can be described as the following five equations, where the subscript :math:`\cdot_{k}`
    is omited for simplicity.

    where superscript :math:`\cdot^{-}` and :math:`\cdot^{+}` denote the priori and
    posteriori estimation, respectively.

    Warning:
        Don't introduce noise in ``System`` methods ``state_transition`` and ``observation``
        for filter testing, as those methods are used for automatically linearizing the system
        by the parent class ``pypose.module.NLS``, unless your system model explicitly
        introduces noise.

    '''
    def __init__(self, model: pp.module.NLS, Q=None, R=None):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model

    def forward(self, state, obs, input, P, dt, Q=None, R=None, t=None):
        r'''
        Performs one step estimation.

        Args:
            state (:obj:`Tensor`): estimated system state of previous step.
            obs (:obj:`Tensor`): system observation at current step (measurement).
            u (:obj:`Tensor`): system input at current step.
            P (:obj:`Tensor`): state estimation covariance of previous step.
            Q (:obj:`Tensor`, optional): covariance of system transition model. Default: ``None``
            R (:obj:`Tensor`, optional): covariance of system observation model. Default: ``None``
            t (:obj:`Tensor`, optional): timestep of system (only for time variant system).
                Default: ``None``

        Return:
            list of :obj:`Tensor`: posteriori state and covariance estimation
        '''
        # Upper cases are matrices, lower cases are vectors
        self.model.set_refpoint(state=state, input=input, dt=dt, t=t)
        I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        c1, c2 = self.model.c1, self.model.c2
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R
        
        xp = self.model.state_transition(state, input, dt, t=t)        # 1. System transition
        # xp = bmv(A, state) + bmv(B, input) + c1        # 1. System transition
        # P = A @ P @ A.mT + Q                  # 2. Covariance predict
        P = A @ P @ A.mT + B @ Q @ B.mT

        K = P @ C.mT @ pinv(C @ P @ C.mT + R) # 3. Kalman gain
        e = obs - bmv(C, state) - bmv(D, input) - c2    #    predicted observation error
        
        xp = xp + bmv(K, e)                     # 4. Posteriori state
        # P = (I - K @ C) @ P                   # 5. Posteriori covariance
        P = (I - K @ C) @ P @ (I - K @ C).mT + K @ R @ K.mT                   # 5. Posteriori covariance
        return xp, P

    def state_propogate(self, state, input, P, dt, Q=None, t=None):
        r'''
        Propogate the system model without observation.
        '''
        self.model.set_refpoint(state=state, input=input, dt=dt, t=t)
        A, B = self.model.A, self.model.B
        xp = self.model.state_transition(state, input, dt, t=t)        # 1. System transition
        # xp = bmv(A, state) + bmv(B, input)         # 1. System transition
        Q = Q if Q is not None else self.Q
        P = A @ P @ A.mT + B @ Q @ B.mT
        return xp, P

    @property
    def Q(self):
        r'''
        The covariance of system transition noise.
        '''
        if not hasattr(self, '_Q'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance Q.')
        return self._Q
    
    @property
    def W(self):
        r'''
        The covariance of system transition noise.
        '''
        if not hasattr(self, '_Q'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance Q.')
        return self._W

    @property
    def R(self):
        r'''
        The covariance of system observation noise.
        '''
        if not hasattr(self, '_R'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance R.')
        return self._R

    def set_uncertainty(self, Q=None, R=None):
        r'''
        Set the covariance matrices of transition noise and observation noise.

        Args:
            Q (:obj:`Tensor`): batched square covariance matrices of transition noise.
            R (:obj:`Tensor`): batched square covariance matrices of observation noise.
        '''
        if Q is not None:
            self.register_buffer("_Q", Q)
        if R is not None:
            self.register_buffer("_R", R)

