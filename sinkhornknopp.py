import torch
import math
import torch.nn.functional as F
from backend import get_backend
import numpy as np


def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]


def sinkhorn_knopp(method, a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True, u=None, v=None, h=None, reg2=1, log_alpha=10, Hy='H1',
                   **kwargs):
    h = h.to('cuda')
    num_class = M.shape[1]
    num_samples = M.shape[0]
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}


    if method == 'RSTC':
        if u == None or v == None:
            if n_hists:
                u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
                v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
            else:
                u = nx.ones(dim_a, type_as=M) / dim_a
                v = nx.ones(dim_b, type_as=M) / dim_b

        K = nx.exp(M / (-reg))

        Kp = (1 / a).reshape(-1, 1) * K

        err = 1

        ######
        h.requires_grad_()
        ########========---------------
        for ii in range(numItermax):
            uprev = u
            vprev = v
            KtransposeU = nx.dot(K.T, u)
            v = b.detach() / KtransposeU
            u = 1. / nx.dot(Kp, v)

            if (nx.any(KtransposeU == 0)
                    or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                    or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration %d' % ii)
                u = uprev
                v = vprev
                break
            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if n_hists:
                    tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
                else:
                    # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                    tmp2 = nx.einsum('i,ij,j->j', u, K, v)
                err = nx.norm(tmp2 - b.detach())  # violation of marginal
                if log:
                    log['err'].append(err)

                if err < stopThr:
                    break
                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))

            # if step >= 0:  # 每一步都更新 0.819 2步更新一次 0.818 b的更新与p的正确与否有关，所以是不是等一些step以后再开始更新b 0.7454(有对比学习+swap predict)
            if Hy == 'H1':
                # if objective == 'saot' or objective == 'all':
                if True:
                    tmpv = v.clone().detach()
                    g = reg * torch.log(tmpv)
                    bfh = 1000
                    for i in range(10):  #
                        g[g == h] += 1e-5  # 处理b是nan的问题
                        delta = ((g - h)) ** 2 + 4 * (reg2 ** 2)  # cluster_num * 1
                        sqrt_delta = torch.sqrt(delta)
                        b = (((g - h) + 2 * reg2) - sqrt_delta) / (2 * (g - h))

                        fh = torch.sum(b) - 1
                        fh.backward()
                        h.data.sub_(fh.data / h.grad.data)  # 1 0.819 0.5 0.816 2 0.818
                        fh = torch.abs(fh)
                        if fh < bfh:
                            bfh = fh
                        else:
                            break
            elif Hy == 'H3':
                # update b H_3
                g = -reg * torch.log(v)
                b = torch.mul(b, torch.exp(g / reg2)) / torch.matmul(b, torch.exp(g / reg2))
            else:
                pass
        else:
            if warn:
                print("Sinkhorn did not converge. You might want to "
                      "increase the number of iterations `numItermax` "
                      "or the regularization parameter `reg`.")

        if log:
            log['niter'] = ii
            log['u'] = u
            log['v'] = v
            log['b'] = b.detach()

        if n_hists:  # return only loss
            res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
            if log:
                return res, log
            else:
                return res

        else:  # return OT matrix

            if log:
                return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
            else:
                return u.reshape((-1, 1)) * K * v.reshape((1, -1))
    elif method == 'entropy':
        lambd1 = reg
        lambd2 = reg2
        if u == None or v == None:
            if n_hists:
                u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
                v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
            else:
                f = (nx.ones(dim_a, type_as=M) / dim_a).reshape(-1, 1)  # 这
                g = (nx.ones(dim_b, type_as=M) / dim_b).reshape(1, -1)
        a = a.reshape(-1, 1)
        b = b.reshape(1, -1)

        err = 1

        ######
        # h.requires_grad_()
        ########========---------------
        for ii in range(numItermax):
            x1 = (g.repeat(num_samples, 1) - M) / lambd1
            max_x1, max_idx1 = torch.max(x1, dim=1)
            max_x1 = max_x1.reshape(-1, 1)
            x1 = x1 - max_x1.repeat(1, num_class)
            exp_x1 = torch.exp(x1)
            f = lambd1 * (torch.log(a) - max_x1) - lambd1 * torch.log(torch.sum(exp_x1, dim=1).reshape(-1, 1))  # 400*1

            x2 = (f.repeat(1, num_class) - M) / lambd1
            max_x2, max_idx2 = torch.max(x2, dim=0)
            max_x2 = max_x2.reshape(1, -1)
            x2 = x2 - max_x2.repeat(num_samples, 1)
            exp_x2 = torch.exp(x2)
            # KL散度
            # g = lambd3 * ((h - g) / lambd2 -1 - max_x2 + torch.log(torch.tensor(1 / num_class))) - lambd3 * torch.log(torch.sum(exp_x2, dim=0).reshape(1, -1))  # 1*20
            # 熵正则化
            g = lambd1 * ((h - g) / lambd2 - 1 - max_x2) - lambd1 * torch.log(
                torch.sum(exp_x2, dim=0).reshape(1, -1))  # 1*20

            # KL散度
            # x3 = torch.exp((g + lambd2) / -lambd2) * (1 / num_class)
            # h = -lambd2 * torch.log(torch.sum(x3))
            # 熵正则化
            x3 = torch.exp((g + lambd2) / -lambd2)
            h = -lambd2 * torch.log(torch.sum(x3))

            if (nx.any(nx.isnan(g)) or nx.any(nx.isnan(f))
                    or nx.any(nx.isinf(g)) or nx.any(nx.isinf(f))
                    or torch.isnan(h) or torch.isinf(h)):
                print('Warning: g or f or h numerical errors at iteration %d' % ii)
                break
            if ii % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if n_hists:
                    # tmp2 = nx.einsum('ik,ij,jk->jk', u, Km, v)
                    tmp2 = 0
                else:
                    # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                    # tmp2 = nx.einsum('i,ij,j->j', u, Km, v)
                    f_matrix = f.repeat(1, num_class)
                    g_matrix = g.repeat(num_samples, 1)
                    Q = torch.exp((f_matrix + g_matrix - M) / lambd1)
                    tmp2 = torch.sum(Q, dim=0)
                err = nx.norm(tmp2 - b.detach())  # violation of marginal
                if log:
                    log['err'].append(err)

                if err < stopThr:
                    break
                if verbose:
                    if ii % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(ii, err))


            elif Hy == 'H3':
                # update b H_3
                g = -lambd1 * torch.log(v)
                b = torch.mul(b, torch.exp(g / lambd2)) / torch.matmul(b, torch.exp(g / lambd2))
            else:
                pass
        else:
            if warn:
                print("Sinkhorn did not converge. You might want to "
                      "increase the number of iterations `numItermax` "
                      "or the regularization parameter `reg`.")

        if log:
            log['niter'] = ii
            log['u'] = u
            log['v'] = v
            log['b'] = b.detach()

        if n_hists:  # return only loss
            res = 0
            if log:
                return res, log
            else:
                return res

        else:  # return OT matrix
            f_matrix = f.repeat(1, num_class)
            g_matrix = g.repeat(num_samples, 1)
            Q = torch.exp((f_matrix + g_matrix - M) / lambd1)
            if log:
                return Q, log
            else:
                return Q