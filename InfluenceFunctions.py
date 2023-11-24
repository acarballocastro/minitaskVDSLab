import abc
import os
from abc import ABC
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.itertools import product

from lfxai.explanations.examples import ExampleBasedExplainer

from lfxai.utils.influence import hessian_vector_product, stack_torch_tensors

class InfluenceFunctions(ExampleBasedExplainer, ABC):
    def __init__(
        self,
        model: nn.Module,
        loss_f: callable,
        save_dir: Path,
        X_train: torch.Tensor = None,
    ):
        super().__init__(model, X_train, loss_f)
        self.save_dir = save_dir
        if not save_dir.exists():
            os.makedirs(save_dir)

    def attribute(
        self,
        X_test: torch.Tensor,
        train_idx: list,
        batch_size: int = 1,
        damp: float = 1e-3,
        scale: float = 1000,
        recursion_depth: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Code adapted from https://github.com/ahmedmalaa/torch-influence-functions/
        This function applies the stochastic estimation approach to evaluating influence function based on the power-series
        approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:
        H^-1 = \\sum^\\infty_{i=0} (I - H) ^ j
        This series converges if all the eigen values of H are less than 1.

        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.
        """

        SUBSAMPLES = batch_size
        NUM_SAMPLES = self.X_train.shape[0]

        loss = [
            self.loss_f(
                self.X_train[idx : idx + 1], self.model(self.X_train[idx : idx + 1])
            )
            for idx in train_idx
        ]

        grads = [
            stack_torch_tensors(
                torch.autograd.grad(
                    loss[k], self.model.encoder.parameters(), create_graph=True
                )
            )
            for k in range(len(train_idx))
        ]

        IHVP_ = [grads[k].detach().cpu().clone() for k in range(len(train_idx))]

        for _ in tqdm(range(recursion_depth), unit="recursion", leave=False):
            sampled_idx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)
            sampled_loss = self.loss_f(
                self.X_train[sampled_idx], self.model(self.X_train[sampled_idx])
            )
            IHVP_prev = [IHVP_[k].detach().clone() for k in range(len(train_idx))]
            hvps_ = [
                stack_torch_tensors(
                    hessian_vector_product(sampled_loss, self.model, [IHVP_prev[k]])
                )
                for k in tqdm(range(len(train_idx)), unit="example", leave=False)
            ]
            IHVP_ = [
                g_ + (1 - damp) * ihvp_ - hvp_ / scale
                for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)
            ]

        IHVP_ = [
            IHVP_[k] / (scale * NUM_SAMPLES) for k in range(len(train_idx))
        ]  # Rescale Hessian-Vector products
        IHVP_ = torch.stack(IHVP_, dim=0).reshape(
            (len(train_idx), -1)
        )  # Make a tensor (len(train_idx), n_params)

        test_loss = [
            self.loss_f(X_test[idx : idx + 1], self.model(X_test[idx : idx + 1]))
            for idx in range(len(X_test))
        ]
        test_grads = [
            stack_torch_tensors(
                torch.autograd.grad(
                    test_loss[k], self.model.encoder.parameters(), create_graph=True
                )
            )
            for k in range(len(X_test))
        ]
        test_grads = torch.stack(test_grads, dim=0).reshape((len(X_test), -1))
        attribution = torch.einsum("ab,cb->ac", test_grads, IHVP_)
        return attribution

    def attribute_loader(
        self,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_loader_replacement: DataLoader,
        recursion_depth: int = 100,
        damp: float = 1e-3,
        scale: float = 1000,
        **kwargs,
    ) -> torch.Tensor:
        attribution = torch.zeros((len(test_loader), len(train_loader)))
        for test_idx, (x_test, _) in enumerate(
            tqdm(test_loader, unit="example", leave=False)
        ):
            x_test = x_test.to(device)
            test_loss = self.loss_f(x_test, self.model(x_test))
            test_grad = (
                stack_torch_tensors(
                    torch.autograd.grad(
                        test_loss, self.model.encoder.parameters(), create_graph=True
                    )
                )
                .detach()
                .clone()
            )
            torch.save(
                test_grad.detach().cpu(), self.save_dir / f"test_grad{test_idx}.pt"
            )
        for train_idx, (x_train, _) in enumerate(
            tqdm(train_loader, unit="example", leave=False)
        ):
            x_train = x_train.to(device)
            loss = self.loss_f(x_train, self.model(x_train))
            grad = stack_torch_tensors(
                torch.autograd.grad(
                    loss, self.model.encoder.parameters(), create_graph=True
                )
            )
            ihvp = grad.detach().clone()
            train_sampler = iter(train_loader_replacement)
            for _ in tqdm(range(recursion_depth), unit="recursion", leave=False):
                X_sample, _ = next(train_sampler)
                X_sample = X_sample.to(device)
                sampled_loss = self.loss_f(X_sample, self.model(X_sample))
                ihvp_prev = ihvp.detach().clone()
                hvp = stack_torch_tensors(
                    hessian_vector_product(sampled_loss, self.model, ihvp_prev)
                )
                ihvp = grad + (1 - damp) * ihvp - hvp / scale
            ihvp = ihvp / (scale * len(train_loader))  # Rescale Hessian-Vector products
            torch.save(ihvp.detach().cpu(), self.save_dir / f"train_ihvp{train_idx}.pt")

        for test_idx, train_idx in product(
            range(len(test_loader)), range(len(train_loader)), leave=False
        ):
            ihvp = torch.load(self.save_dir / f"train_ihvp{train_idx}.pt")
            test_grad = torch.load(self.save_dir / f"test_grad{test_idx}.pt")
            attribution[test_idx, train_idx] = (
                torch.dot(test_grad.flatten(), ihvp.flatten()).detach().clone().cpu()
            )
        return attribution

    def __str__(self):
        return "Influence Functions"