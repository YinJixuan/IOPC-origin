import torch
# from solver import *
import torch.nn.functional as F
import time
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.bregman import sinkhorn
import ot
import time
import warnings
import sinkhornknopp as sk
from torch.nn.functional import normalize

def entropic_COT_extra_reg(method, a, b, M, lambda2, lambda3, f, df, G0=None, numItermax=10,
                                    numInnerItermax=200, stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, version='fast'):
    coupling = entropic_COT_gcg(method, a, b, M, lambda2, lambda3, f, df, G0,
                                        numItermax, numInnerItermax, stopThr, stopThr2, 
                                        verbose, log, version=version)
    return coupling

def entropic_COT_gcg(method, a, b, M, lambda2, lambda3, f, df, Q0=None, numItermax=10,
        numInnerItermax=200, stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, version='fast'):
    r"""
    modify from ot.optim.gcg in the direction finding part with entropic_partial_wasserstein solver
    ot.optim.gcg: https://pythonot.github.io/_modules/ot/partial.html#partial_gromov_wasserstein

    """
    a, b, M, Q0 = list_to_array(a, b, M, Q0)
    nx = get_backend(a, M)

    loop = 1

    if log:
        log = {'loss': []}
    if Q0 is None:
        Q = nx.outer(a, b)
    else:
        Q = Q0

    def cost(Q):
        if method == 'RSTC':
            p_ki = torch.sum(Q, dim=0)
            struc_b = torch.sum(-torch.log(p_ki) - torch.log(1 - p_ki))
            return nx.sum(M * Q) + f(Q) + lambda2 * struc_b + lambda3 * nx.sum(Q * nx.log(Q) - Q)
        elif method == 'entropy':
            p_ki = torch.sum(Q, dim=0)
            struc_b = - torch.sum(p_ki * torch.log(p_ki))
            return nx.sum(M * Q) + f(Q) + lambda2 * struc_b + lambda3 * nx.sum(Q * nx.log(Q) - Q)

    f_val = cost(Q)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val.item(), 0, 0))
    
    if version == 'normal':
        func = entropic_COT
    elif version == 'fast':
        func = entropic_COT_fast
    while loop:
        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + df(Q)

        Qc = sk.sinkhorn_knopp(method, a, b, Mi, lambda3, numItermax=numItermax, warn=False, log=False,
                               u=None, v=None, h=torch.FloatTensor([1]), reg2=lambda2, log_alpha=10,
                               Hy='H1')

        if torch.any(torch.isnan(Qc)) or torch.any(torch.isinf(Qc)):
            print('Warning: numerical errors at iteration', it)
            break
        deltaQ = Qc - Q

        # line search
        dcost = Mi + lambda3 * (nx.log(Q))  # ??    # 目标函数的导数
        alpha, fc, f_val = line_search_armijo(
            cost, Q, deltaQ, dcost, f_val, alpha_min=0., alpha_max=1.   # 目标函数，当前成本，前进方向，目标函数的导数，目标函数的数值
        )

        Q = Q + alpha * deltaQ

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)

        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val.item(), relative_delta_fval.item(), abs_delta_fval.item()))

    if log:
        return Q, log
    else:
        return Q

def entropic_COT(a, b, M, reg, numItermax=1000,
                            stopThr=1e-9, verbose=False, log=False):
    r"""
    modify from ot.partial.entropic_partial_wasserstein in torch version

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = a.to(torch.float64)
    b = b.to(torch.float64)
    M = M.to(torch.float64)

    dim_a, dim_b = M.shape
    dx = torch.ones(dim_a, dtype=torch.float64).to(device)
    dy = torch.ones(dim_b, dtype=torch.float64).to(device)

    log_e = {'err': []}

    K = torch.exp(M / (-reg))

    err, cpt = 1, 0

    q1 = torch.ones(K.shape).to(device)
    q2 = torch.ones(K.shape).to(device)
    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K = K * q1
        K1 = torch.matmul(torch.diag(torch.minimum(a / torch.sum(K, axis=1), dx)), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        K = torch.matmul(K1, torch.diag(b / torch.sum(K1, axis=0)))
        q2 = q2 * K1prev / K

        cpt = cpt + 1
    if log:
        return K, log_e
    else:
        return K


def entropic_COT_fast(a, b, M, reg, numItermax=1000,
                            stopThr=1e-9, verbose=False, log=False):
    r"""
    modify from ot.partial.entropic_partial_wasserstein in torch version

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = a.to(torch.float64)
    b = b.to(torch.float64)
    M = M.to(torch.float64)

    dim_a, dim_b = M.shape
    dx = torch.ones(dim_a, dtype=torch.float64).to(device)
    dy = torch.ones(dim_b, dtype=torch.float64).to(device)

    log_e = {'err': []}

    K = torch.exp(M / (-reg))

    Kp = torch.matmul(torch.diag(1 / a), K)
    Kq = torch.matmul(torch.diag(1 / b), K.T)

    err, cpt = 1, 0
    u = dx
    v = dy
    while (cpt < numItermax):
        # temp_u = u
        # temp_v = v

        u = torch.div(dx, torch.matmul(Kp, v))
        # u = torch.minimum(temp, dx)
        v = torch.div(dy, torch.matmul(Kq, u))

        cpt = cpt + 1
    Kprev = torch.matmul(torch.diag(u), K)
    Kprev = torch.matmul(Kprev, torch.diag(v))
    if log:
        return Kprev, log_e
    else:
        return Kprev
    

def line_search_armijo(
    f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
    alpha0=0.99, alpha_min=None, alpha_max=None
):
    r"""
    modify from ot.optim.line_search_armijo

    """
    xk, pk, gfk = list_to_array(xk, pk, gfk)
    nx = get_backend(xk, pk)

    if len(xk.shape) == 0:
        xk = nx.reshape(xk, (-1,))

    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = nx.sum(pk * gfk)  # Quickfix for matrices
    # tt = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    if alpha is None:
        return 0., fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
            # if type(alpha) == torch.Tensor:
            #     alpha = float(alpha.item())
            # alpha = torch.clip(alpha, alpha_min, alpha_max)

            if type(alpha) != torch.Tensor:
                alpha = torch.tensor(alpha)
            alpha = torch.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1
    
def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """
    modify from andoptimize.linesearch.scalar_search_armijo

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    return None, phi_a1

def curriculum_structure_aware_PL(method, a, b, P, feature, lambda1=0.1, lambda2=0.1, lambda3=0.1, version='fast', reg_e=0.01, reg_sparsity=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # P_normal = F.normalize(P, dim=1)
    # S = torch.matmul(P_normal, P_normal.t()).to(device)

    S = torch.matmul(feature, feature.t()).to(device)
    mask = torch.eye(S.size(0), S.size(0)).bool().to(device)
    S.masked_fill_(mask, 0)
    S = S.to(torch.float64)
    P = P.to(device)
    cost = -torch.log(P)


    def structure_aware_f(Q):
        # 语义约束
        temp1 = torch.matmul(Q, Q.t())
        temp2 = torch.mul(S, temp1)
        struc_sem = torch.sum(temp2)

        res = -lambda1 * struc_sem
        return res

    def structure_aware_df(Q):
        # 语义约束的梯度
        gradient_sem = torch.matmul(S, Q) + torch.matmul(S.t(), Q)

        gradient = -lambda1 * gradient_sem
        return gradient


    coupling = entropic_COT_extra_reg(method, a, b, M=cost, lambda2=lambda2, lambda3=lambda3,
                                        f=structure_aware_f, df=structure_aware_df,
                                        G0=None, numItermax=10, numInnerItermax=100, stopThr=1e-6, stopThr2=1e-6, 
                                        verbose=False, log=False, version=version)
    pseudo_labels = torch.argmax(coupling, dim=1)
    current_b = torch.sum(coupling, dim=0)

    # probs = F.softmax(coupling, dim=1)
    max_probs, _ = torch.max(coupling, dim=1)

    return pseudo_labels, current_b, max_probs

def omega(S, M, Q):
    temp = torch.mul(M, Q)
    temp1 = torch.matmul(temp, temp.t())
    temp2 = torch.mul(S, temp1)
    res = torch.sum(temp2)
    return res

def omega_df(S, M, Q):
    temp = torch.mul(M, Q)
    gradient = 2 * torch.mul(torch.matmul(S, temp), M)
    return gradient

