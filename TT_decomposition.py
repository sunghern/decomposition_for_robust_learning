'''
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from joblib import Parallel, delayed
from multiprocessing import Process, Manager, cpu_count, Pool

class SolveIndividual:
    def solve(self, A, b, nu, rho, Z):
        t1 = A.dot(A.T)
        A = A.reshape(-1, 1)
        tX = (A * b + rho * Z - nu) / (t1 + rho)
        return tX
'''

import os
from re import S
import torch 
import torchtt # torchTT for custom
import pytest
import argparse
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from numpy.linalg import inv
from numpy.linalg import norm
from joblib import Parallel, delayed
from multiprocessing import Process, Manager, cpu_count, Pool
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Decomposition for Robust Learning')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--train-dataset', default='cifar10', type=str)
parser.add_argument('--test-dataset', default='cifar10', type=str)
parser.add_argument('--test-decomposition-method', default='tensor_train', typr=str)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argumnet('--checkpath', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--tensor-core-init', default=None, type=str)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--warmup-lr', default=0, type=float)

use_cuda = torch.cuda.is_available()

def set_random_seed(seed=None):
  if seed == None:
    seed = 0
  print('seed for random sampling: {}', format(seed))
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def cosine_calc_learning_rate(args, epoch, batch=0, nBatch=None):
  T_total = args.epochs * nBatch
  T_cur = epoch * nBatch + batch
  lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
  return lr

def cosine_adjust_learning_rate(args, optimizer, epoch, batch=0, nBatch=None):
  new_lr = cosine_calc_learning_rate(args, epoch, batch, nBatch)
  for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr
  return new_lr

def cosine_warmup_adjust_learning_rate(args, optimizer, T_total, nBatch,
                                      epoch, batch=0, warmup_lr=0):
  T_cur = epoch * nBatch + batch + 1
  new_lr = T_cur / T_total * (args.lr - warmup_lr) + warmup_lr
  for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr
  return new_lr               

def train(train_loader, model, criterion, soft_criterion, optimizer, epoch, args):
  print('train dataset :', args.train_dataset)

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  # progress = ProgressMeter()

  # switch to train mode
  model.train()
  # pretrained.eval()

  for i, (images, target) in enumerate(train_loader):
    if args.lr_scheduler == 'cosine':
      nBatch = len(train_loader)
      if epoch < args.warmup_epochs:
        cosine_warmup_adjust_learning_rate(args, optimizer,
        args.warmup_epochs * nBatch, nBatch, epoch, i, args.warmup_lr)
      else:
        cosine_adjust_learning_rate(
          args, optimizer, epoch - args.warmup_epochs, i, nBatch)
    
  output = model(images)
  loss = criterion(output, target)

  acc1, acc5 = accuracy(output, target, topk=(1, 5))
  losses.update(loss.item(), images.size(0))
  top1.update(acc1[0], images.size(0))
  top5.update(acc5[0], images.size(0))
  loss.backward()

  # compute gradient
  optimizer.step()


def validation(val_loader, model, criterion, args, epoch, test_dataset):
  print('validation datset :', args.test_dataset)
  print('decomposition method :', args.test_decomposition_method)


""""""""""""" code for Tensor Train decomposition """""""""""""

def err_rel(t, ref): return tn.linalg.norm(t-ref).numpy() / \
    torch.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf

parameters = [torch.float64, torch.complex128]

@pytest.mark.parametrize("dtype", parameters)
def test_init(dtype):
    """
    Checks the constructor and the TT.full() function. 
    A list of cores is passed and is checked if the recomposed tensor is correct.
    """
    # print('Testing: Initialization from list of cores.')
    cores = [torch.rand([1, 20, 3], dtype=dtype), torch.rand(
        [3, 10, 4], dtype=dtype), torch.rand([4, 5, 1], dtype=dtype)]

    T = torchtt.TT(cores)

    Tfull = T.full()
    T_ref = torch.squeeze(torch.einsum('ijk,klm,mno->ijlno',
                       cores[0], cores[1], cores[2]))

    assert err_rel(Tfull, T_ref) < 1e-14

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_random(dtype):
    """
    Perform a TT decomposition of a random full random tensor and check if the decomposition is accurate.
    """
    # print('Testing: TT-decomposition from full (random tensor).')
    T_ref = torch.rand([10, 20, 30, 5], dtype=dtype)

    T = torchtt.TT(T_ref, eps=1e-19, rmax=1000)

    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_lowrank(dtype):
    """
    Check the decomposition of a tensor which is already in the low rank format.
    """
    # print('Testing: TT-decomposition from full (already low-rank).')
    cores = [torch.rand([1, 200, 30], dtype=dtype), torch.rand(
        [30, 100, 4], dtype=dtype), torch.rand([4, 50, 1], dtype=dtype)]
    T_ref = torch.squeeze(torch.einsum('ijk,klm,mno->ijlno',
                       cores[0], cores[1], cores[2]))

    T = torchtt.TT(T_ref, eps=1e-19)

    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_highd(dtype):
    """
    Decompose a 20d tensor with all modes 2.

    Returns
    -------
    None.

    """
    # print('Testing: TT-decomposition from full (long  20d TT).')
    cores = [torch.rand([1, 2, 16], dtype=dtype)] + [torch.rand([16, 2, 16], dtype=dtype)
                                                  for i in range(18)] + [torch.rand([16, 2, 1], dtype=dtype)]
    T_ref = torchtt.TT(cores).full()

    T = torchtt.TT(T_ref, eps=1e-12)
    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_ttm(dtype):
    """
    Decompose a TT-matrix.

    Returns
    -------
    bool
        True if test is passed.

    """

    T_ref = torch.rand([10, 11, 12, 15, 17, 19], dtype=dtype)

    T = torchtt.TT(T_ref, shape=[(10, 15), (11, 17),
                (12, 19)], eps=1e-19, rmax=1000)
    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_orthogonal(dtype):
    """
    Checks the lr_orthogonal function. The reconstructed tensor should remain the same.
    """
    # print('Testing: TT-orthogonalization.')
    cores = [torch.rand([1, 20, 3], dtype=dtype), torch.rand([3, 10, 4], dtype=dtype), torch.rand(
        [4, 5, 20], dtype=dtype), torch.rand([20, 5, 2], dtype=dtype), torch.rand([2, 10, 1], dtype=dtype)]
    T = torchtt.TT(cores)
    T = torchtt.random([3, 4, 5, 3, 8, 7, 10, 3, 5, 6], [
                    1, 20, 12, 34, 3, 50, 100, 12, 2, 80, 1], dtype=dtype)
    T_ref = T.full()

    cores, R = torchtt._decomposition.lr_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = torchtt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Left to right ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        L = torch.reshape(c, [-1, c.shape[-1]]).numpy()
        assert np.linalg.norm(L.T @ np.conj(L) - np.eye(L.shape[1])) < 1e-12 or i == len(
            cores)-1, 'Cores are not left orthogonal after LR orthogonalization.'

    cores, R = torchtt._decomposition.rl_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = torchtt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Right to left ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        R = torch.reshape(c, [c.shape[0], -1]).numpy()
        assert np.linalg.norm(
            np.conj(R) @ R.T - np.eye(R.shape[0])) < 1e-12 or i == 0

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_orthogonal_ttm(dtype):
    """
    Test the lr and rt orthogonal functions for a TT matrix.
    """
    T = torchtt.random([(3, 4), (5, 6), (7, 8), (9, 4)],
                    [1, 2, 3, 4, 1], dtype=dtype)
    T_ref = T.full()

    cores, R = torchtt._decomposition.lr_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = torchtt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Left to right ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        L = torch.reshape(c, [-1, c.shape[-1]]).numpy()
        assert np.linalg.norm(L.T @ np.conj(L) - np.eye(L.shape[1])) < 1e-12 or i == len(
            cores)-1, 'Cores are not left orthogonal after LR orthogonalization.'

    cores, R = torchtt._decomposition.rl_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = torchtt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Right to left ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        R = torch.reshape(c, [c.shape[0], -1]).numpy()
        assert np.linalg.norm(
            np.conj(R) @ R.T - np.eye(R.shape[0])) < 1e-12 or i == 0

@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_rounding(dtype):
    """
    Testing the rounding of a TT-tensor.
    A rank-4tensor is constructed and successive approximations are performed.
    """
    # print('Testing: TT-rounding.')

    T1 = torch.einsum('i,j,k->ijk', torch.rand([20], dtype=dtype),
                   torch.rand([30], dtype=dtype), torch.rand([32], dtype=dtype))
    T2 = torch.einsum('i,j,k->ijk', torch.rand([20], dtype=dtype),
                   torch.rand([30], dtype=dtype), torch.rand([32], dtype=dtype))
    T3 = torch.einsum('i,j,k->ijk', torch.rand([20], dtype=dtype),
                   torch.rand([30], dtype=dtype), torch.rand([32], dtype=dtype))
    T4 = torch.einsum('i,j,k->ijk', torch.rand([20], dtype=dtype),
                   torch.rand([30], dtype=dtype), torch.rand([32], dtype=dtype))

    T_ref = T1 / torch.linalg.norm(T1) + 1e-3*T2 / torch.linalg.norm(T2) + \
        1e-6*T3 / torch.linalg.norm(T3) + 1e-9*T4 / torch.linalg.norm(T4)
    T3 = T1 / torch.linalg.norm(T1) + 1e-3*T2 / \
        torch.linalg.norm(T2) + 1e-6*T3 / torch.linalg.norm(T3)
    T2 = T1 / torch.linalg.norm(T1) + 1e-3*T2 / torch.linalg.norm(T2)
    T1 = T1 / torch.linalg.norm(T1)

    T = torchtt.TT(T_ref)
    T = T.round(1e-9)
    Tfull = T.full()
    assert T.R == [1, 3, 3, 1], 'Case 1: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-9, 'Case 1: error too high'

    T = torchtt.TT(T_ref)
    T = T.round(1e-6)
    Tfull = T.full()
    assert T.R == [1, 2, 2, 1], 'Case 2: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-6, 'Case 1: error too high'

    T = torchtt.TT(T_ref)
    T = T.round(1e-3)
    Tfull = T.full()
    assert T.R == [1, 1, 1, 1], 'Case 3: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-3, 'Case 1: error too high'

@pytest.mark.parametrize("dtype", parameters)
def test_dimension_permute(dtype):
    """
    Test the permute function.
    """
    x_tt = torchtt.random([5, 6, 7, 8, 9], [1, 2, 3, 4, 2, 1])
    x_ref = x_tt.full()
    xp_tt = torchtt.permute(x_tt, [4, 3, 2, 1, 0], 1e-10)
    xp_ref = torch.permute(x_ref, [4, 3, 2, 1, 0])

    assert tuple(xp_tt.N) == tuple(
        xp_ref.shape), 'Permute modex of a TT tensor: shape mismatch.'
    assert err_rel(
        xp_tt.full(), xp_ref) < 1e-10, 'Permute modex of a TT tensor: error too high.'

    # Test for TT matrices
    A_tt = torchtt.random([(2, 3), (4, 5), (3, 2), (6, 7),
                       (5, 3)], [1, 2, 3, 4, 2, 1])
    A_ref = A_tt.full()
    Ap_tt = torchtt.permute(A_tt, [3, 2, 4, 0, 1])
    Ap_ref = torch.permute(A_ref, [3, 2, 4, 0, 1, 8, 7, 9, 5, 6])

    assert Ap_tt.M == [6, 3, 5, 2,
                       4], 'Permute modex of a TT matrix: shape mismatch.'
    assert Ap_tt.N == [7, 2, 3, 3,
                       5], 'Permute modex of a TT matrix: shape mismatch.'
    assert err_rel(
        Ap_tt.full(), Ap_ref) < 1e-10, 'Permute modex of a TT tensor: error too high.'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def TR_decomposition():
  print('TR decomposition')

def Tucker_decomposition():
  print('Tucker decomposition')

def CP_decomposition():
  print('CP decomposition')

def SVD():
  print('SVD')

""""""""""""" code for ADMM """""""""""""

class SolveIndividual:
  def solve(self, A, b, nu, rho, Z): 
    t1 = A.dot(A.T)
    A = A.reshape(-1, 1)
    tX = (A * b + rho * Z - nu) / (t1 + rho)
    return tX

class CombineSolution:
  def combine(self, nuBar, xBar, Z, rho):
    t = nuBar.reshape(-1, 1)
    t = t + rho * (xBar.reshape(-1, 1) - Z)
    return t.T

class ADMM:
    def __init__(self, A, b, parallel = False):
        self.D = A.shape[1]
        self.N = A.shape[0]
        if parallel:
            self.XBar = np.zeros((self.N, self.D))
            self.nuBar = np.zeros((self.N, self.D))
        self.nu = np.zeros((self.D, 1))
        self.rho = 1
        self.X = np.random.randn(self.D, 1)
        self.Z = np.zeros((self.D, 1))
        self.A = A
        self.b = b
        self.alpha = 0.01
        self.parallel = parallel
        self.numberOfThreads = cpu_count()

    def step(self):
        if self.parallel:
            return self.step_parallel()

        # Solve for X_t+1
        self.X = inv(self.A.T.dot(self.A) + self.rho).dot(self.A.T.dot(self.b) + self.rho * self.Z - self.nu)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) *  np.sign(self.Z)
        # Combine
        self.nu = self.nu + self.rho * (self.X - self.Z)

    def solveIndividual(self, i):
        solve = SolveIndividual()
        return solve.solve(self.A[i], np.asscalar(self.b[i]), self.nuBar[i].reshape(-1, 1), self.rho, self.Z)

    def combineSolution(self, i):
        combine = CombineSolution()
        return combine.combine(self.nuBar[i].reshape(-1, 1), self.XBar[i].reshape(-1, 1), self.Z, self.rho)

    def step_parallel(self):
        # Solve for X_t+1
        #Parallel(n_jobs = self.numberOfThreads, backend = "threading")(
        #    delayed(self.solveIndividual)(i) for i in range(0, self.N-1))
        process = []
        for i in range(0, self.N-1):
            p = Process(target = self.solveIndividual, args= (i,))
            p.start()
            process.append(p)

        for p in process:
            p.join()

        self.X = np.average(self.XBar, axis = 0)
        self.nu = np.average(self.nuBar, axis = 0)

        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * np.sign(self.Z)
        # Combine
        #Parallel(n_jobs = self.numberOfThreads, backend = "threading")(
        #    delayed(self.combineSolution)(i) for i in range(0, self.N-1))

        process = []
        for i in range(0, self.N-1):
            p = Process(target = self.combineSolution, args= (i,))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        
    def step_iterative(self):
        # Solve for X_t+1
        for i in range(0, self.N-1):
            t = self.solveIndividual(i)
            self.XBar[i] = t.T

        self.X = np.average(self.XBar, axis = 0)
        self.nu = np.average(self.nuBar, axis = 0)

        self.X = self.X.reshape(-1, 1)
        self.nu = self.nu.reshape(-1, 1)

        # Solve for Z_t+1
        self.Z = self.X + self.nu / self.rho - (self.alpha / self.rho) * np.sign(self.Z)

        # Combine
        for i in range(0, self.N-1):
            t = self.nuBar[i].reshape(-1, 1)
            t = t + self.rho * (self.XBar[i].reshape(-1, 1) - self.Z)
            self.nuBar[i] = t.T

    def LassoObjective(self):
        return 0.5 * norm(self.A.dot(self.X) - self.b)**2 + self.alpha *  norm(self.X, 1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class AverageMeter(object):
  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0 
    self.avg = 0
    self.sum = 0
    self.count = 0;

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '{' + fmt + '/' + fmt.fotmat(num_batches) + '}'

def accuracy(output, target, topk=(1,)):
  """ Compute the accuracy over the k top predictions for the specified valuse"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, Ture)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    
    return res

def set_random_seed(seed=None):
  if seed == None:
    seed = 0
  print('seed for random sampling: {}', format(seed))
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def main():
  args = parser.parser_args()
  print(args.checkpath)
  print("learning rate :", args.lr)
  set_random_seed(args.seed)

  os.makedirs('log', exist_ok=True)
  os.makedirs('ckpts', exist_ok=True)
  log_path = os.path.join('log', args.checkpath + '.log')
  if os.path.isfile(log_path)
    os.remove(log_path)
  logger.add(log_path)

  if args.dataset == 'cifar10':
    num_classes = 10
  elif args.dataset == 'cifar100':
    num_classes = 100
  elif args.dataset == 'imagenet':
    num_classes = 1000
  
  model = models.__dict__[args.arch](num_classes=num_classes)
  model = model.cuda('cuad:{}'.format(args.gpu))

  if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

  # data load
  # train_loader, val_loader = data_loader()

  criterion = torch.nn.CrossEntropyLoss()
  soft_criterion = CrossEntropyLossSoft(reduction="mean")

  # evaluate
  if args.evaluate:
    print('evaluation')

  for epoch in range(args.epochs):
    train(train_loader, model, criterion, soft_criterion, optimizer, epoch
          args)

    acc1 = validate(val_loader, model, criterion, args, epoch)

    if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'ckpts/%s-latest.pth' % args.checkpath, _use_new_zipfile_serialization = False)
            logger.info('saved to ckpts/%s-latest.pth' % args.checkpath)
    if best_accuracy < acc1:
        best_accuracy = acc1
        torch.save(model.state_dict(), 'ckpts/%s-best.pth' % args.checkpath, _use_new_zipfile_serialization = False)
        logger.info('saved to ckpts/%s-best.pth' % args.checkpath)





