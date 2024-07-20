import torch
from tqdm import trange
import numpy as np
import sys
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = int(sys.argv[1])
D = int(sys.argv[2])
prefix = int(sys.argv[3])

TESTNAME = sys.argv[4] if len(sys.argv) > 4 else 'randn'

if TESTNAME == 'ones':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    v = torch.ones((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16)
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    v = torch.randn((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16)
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16).requires_grad_()
    v = (torch.arange(D, dtype=torch.bfloat16)/D).reshape((1,1,1,-1)).repeat(B, H, N, 1).requires_grad_()
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16)
else:
    print('Invalid test name')
    sys.exit(0)

softmax_scale = 1 / math.sqrt(D)
l_vec = torch.einsum("bhnd,bhmd->bhnm", q.clone(), k.clone()) * softmax_scale

mask = torch.triu(torch.ones(l_vec.shape[2], l_vec.shape[3]), diagonal=1).bool().unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
l_vec = l_vec.masked_fill(mask, float('-inf'))

max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec = l_vec.sum(dim=-1, keepdim=True)

l_vec = max_vec + torch.log(l_vec)

attn_mask = torch.triu(torch.ones(k.shape[2], k.shape[2]), diagonal=1)
for col in range(0, prefix):
    for row in range(0, attn_mask.size(dim=1)):
        attn_mask[row][col] = 0
print(attn_mask)
attn_mask = attn_mask.bool().unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
o.backward(grad_output)
q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

d_vec = torch.mul(o, grad_output)
d_vec = d_vec.sum(dim=-1, keepdim=True)

with open(f'{TESTNAME}_causal_{N}N_{D}D_{prefix}prefix.txt', 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    grad_outputf = grad_output.to(torch.float32).flatten().detach().cpu().numpy()
    
    # intermediate
    l_vecf = l_vec.to(torch.float32).flatten().detach().cpu().numpy()
    d_vecf = d_vec.to(torch.float32).flatten().detach().cpu().numpy()
    
    # outputs
    q_grad = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    k_grad = k_grad.to(torch.float32).flatten().detach().cpu().numpy()
    v_grad = v_grad.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*N):
        f.write(repr(l_vecf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(grad_outputf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(q_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(k_grad[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(v_grad[i]))
        f.write(' ')
        

print(f'Run the harness like `cat {TESTNAME}.txt | ./harness`')

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

B = 32
H = 16
N = int(sys.argv[1])
D = int(sys.argv[2])

print("-" * 60)
print(f'Timing forward pass for for B={B}, H={H}, N={N}, D={D}')
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=True,
    enable_mem_efficient=False
):
    q = torch.randn((B, H, N, D), dtype=torch.float16)
    k = torch.randn((B, H, N, D), dtype=torch.float16)
    v = torch.randn((B, H, N, D), dtype=torch.float16)
    
    q.grad = None
    k.grad = None
    v.grad = None


    # warmup
    for _ in range(10):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

    # Time the forward pass

    for i in range(30):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    
print(f'Timing backward pass for for B={B}, H={H}, N={N}, D={D}')
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=True,
    enable_mem_efficient=False
):
    q = torch.randn((B, H, N, D), dtype=torch.float16).requires_grad_()
    k = torch.randn((B, H, N, D), dtype=torch.float16).requires_grad_()
    v = torch.randn((B, H, N, D), dtype=torch.float16).requires_grad_()
    
    grad_output = torch.randn((B, H, N, D), dtype=torch.float16)
    
    # warmup
    for _ in range(10):
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o.backward(grad_output)
    
    
    # Time the backward pass
    
    for i in range(30):
        q.grad = None
        k.grad = None
        v.grad = None
        grad_output.grad = None
        
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
