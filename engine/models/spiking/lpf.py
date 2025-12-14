import torch, pdb
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import os

class LPFOnline(nn.Module):
    def __init__(
        self, 
        num_channels: int,
        kernel_size: int,
        path_to_image: str,
        tau_mem: float = 50,
        tau_syn: float = 50,
        initial_scale: float = 0.012,
        train_scale: bool = False 
    ):
        super().__init__()   
        self.scale_factor = nn.Parameter(torch.tensor(initial_scale),  requires_grad=train_scale)
        self.tau_mem = nn.Parameter(torch.tensor(tau_mem),  requires_grad=False)
        self.tau_syn = nn.Parameter(torch.tensor(tau_syn),  requires_grad=False)
        #kernel = self.set_low_pass_kernel(kernel_size, tau_mem, tau_syn)

        self.kernel_size = kernel_size
        self.path_to_image =path_to_image 
               
        #self.kernel = self.set_custom_kernel()
        self.num_channels = num_channels
        #self.pad_size = self.kernel.shape[-1] - 1
        # self.conv = torch.nn.Conv1d(
        #     self.num_channels,
        #     self.num_channels,
        #     self.kernel.shape[-1],
        #     bias=False,
        #     groups= self.num_channels,
        # )  
        #self.conv.weight.data = kernel.flip(-1).repeat(num_channels, 1, 1)
        #self.conv.weight.requires_grad_(True)
        self.register_buffer("past_inputs", torch.zeros(1))
        #self.plot(0)

    def plot(self, kernel, step):
        plt.plot(kernel.flip(-1).cpu(), 'o')
        plt.gca().invert_xaxis()
        plt.xticks(np.arange(self.kernel_size, -1, -1), np.arange(self.kernel_size, -1, -1).astype(int))
        plt.xlabel('Timesteps in the past', fontsize=16)
        plt.ylabel('Weights', fontsize=16)
        plt.savefig(os.path.join(self.path_to_image, f"{step}.png"))
        plt.close()
     

    def set_custom_kernel(self):

        syn_kernel = (
            torch.exp(-torch.arange(self.kernel_size) / self.tau_syn).unsqueeze(0).unsqueeze(0)
            )
        mem_kernel = (
            torch.exp(-torch.arange(self.kernel_size) / self.tau_mem).unsqueeze(0).unsqueeze(0)
            )

        # "Padding" only at beginning of syn_kernel.
        padding = torch.zeros_like(syn_kernel)
        syn_kernel = torch.cat((padding, syn_kernel), -1) 
        kernel = torch.nn.functional.conv1d(syn_kernel.flip(-1), mem_kernel.flip(-1))[..., :-1]
        return kernel

    def set_low_pass_kernel(self, kernel_size, tau_mem, tau_syn):

        syn_kernel = (
            torch.exp(-torch.arange(kernel_size) / tau_syn).unsqueeze(0).unsqueeze(0)
        )
        mem_kernel = (
            torch.exp(-torch.arange(kernel_size) / tau_mem).unsqueeze(0).unsqueeze(0)
        )

        # "Padding" only at beginning of syn_kernel.
        padding = torch.zeros_like(syn_kernel)
        syn_kernel = torch.cat((padding, syn_kernel), -1)
        kernel = torch.nn.functional.conv1d(syn_kernel, mem_kernel.flip(-1))[..., :-1]
        return kernel


    def shift_past_inputs(self, shift_amount: int, assign: torch.Tensor):
        # Shift the past_inputs by N along the last axis (T)
        self.past_inputs = torch.roll(self.past_inputs, -shift_amount, -1)
        self.past_inputs[..., -shift_amount:] = assign

    def reset_past(self, shape=None, device=None, dtype=None):
        shape = shape or self.past_inputs.shape
        device = device or self.past_inputs.device
        dtype = dtype or self.past_inputs.dtype
        self.past_inputs = torch.zeros(shape, device=device, dtype=dtype)

    def forward(self, x, padding_mode="past"):
        original_shape = x.shape
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, C, T)

        device = x.device
        dtype = x.dtype

        scale = self.scale_factor.to(device=device, dtype=dtype)
        tau_syn = self.tau_syn.to(device=device, dtype=dtype)
        tau_mem = self.tau_mem.to(device=device, dtype=dtype)

        t = torch.arange(self.kernel_size, device=device, dtype=dtype)
        syn_kernel = torch.exp(-t / tau_syn).unsqueeze(0).unsqueeze(0)   # (1,1,K)
        mem_kernel = torch.exp(-t / tau_mem).unsqueeze(0).unsqueeze(0)   # (1,1,K)

        padding = torch.zeros_like(syn_kernel)
        syn_kernel = torch.cat((padding, syn_kernel), -1)                # (1,1,2K)
        kernel = torch.nn.functional.conv1d(syn_kernel.flip(-1), mem_kernel.flip(-1))[..., :-1]
        self.pad_size = kernel.shape[-1] - 1

        # past_inputs 永远保持在同 device/dtype
        if (shape := x.shape[:-1]) != self.past_inputs.shape[:-1] or self.past_inputs.device != device:
            self.reset_past(shape=(*shape, self.pad_size), device=device, dtype=dtype)

        if padding_mode == "past":
            padded = torch.cat((self.past_inputs, x), -1)

            convd = torch.nn.functional.conv1d(
                padded,
                kernel.flip(-1).repeat(self.num_channels, 1, 1),
                groups=self.num_channels,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
            ) * scale

            # 用 detach，不要用 .data
            self.shift_past_inputs(x.shape[-1], x[..., -self.pad_size:].detach())
            return convd.reshape(*original_shape)

        raise NotImplementedError(padding_mode)
  
        # elif padding_mode=="repeat":
        #     padded = torch.cat((x[..., :1].repeat(1, 1, self.pad_size), x), -1) 
        #     convd = torch.nn.functional.conv1d( padded, 
        #                                         kernel.flip(-1).repeat(self.num_channels, 1, 1),
        #                                         groups=self.num_channels, 
        #                                         bias=False) * self.scale_factor 
        
        # elif padding_mode=="zeros":
        #     padded = torch.cat((torch.zeros_like(x[..., :1]).repeat(1, 1, self.pad_size), x), -1) 
        #     convd = torch.nn.functional.conv1d( padded, 
        #                                         kernel.flip(-1).repeat(self.num_channels, 1, 1),
        #                                         groups=self.num_channels, 
        #                                         bias=False) * self.scale_factor 
        # elif padding_mode=="none":  
        #     convd = torch.nn.functional.conv1d( padded, 
        #                                         kernel.flip(-1).repeat(self.num_channels, 1, 1),
        #                                         groups=self.num_channels, 
        #                                         bias=False) * self.scale_factor 

        
