# Efficient 2:4 Sparse Pre-training
This repository provides the official implementations of efficient 2:4 pre-training toolkit from "Accelerating Transformer Pre-training with 2:4 Sparsity".

arxiv: [https://arxiv.org/abs/2404.01847](https://arxiv.org/abs/2404.01847)

For scripts to replicate the experimental results, please visit [https://github.com/thu-ml/2by4-pretrain-acc-examples](https://github.com/thu-ml/2by4-pretrain-acc-examples)

## Usage

### Using 2:4-spMM

To get started with 2:4-spMM, official [torch.sparse](https://pytorch.org/docs/stable/sparse.html#sparse-semi-structured-tensors) works well enough. However, we've added more features on top of that.

```
from sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# convert FloatTensor x to be 2:4 sparse
to_sparse_semi_structured(x, mask=mask, dtype=torch.half)
to_sparse_semi_structured(x, mask=mask, dtype=torch.bfloat16)

# prune and make conversion with self-defined arbitrary 2:4 mask
to_sparse_semi_structured(x, mask=mask, dtype=torch.half)

# MVUE
to_sparse_semi_structured(x, MVUE24=True, dtype=torch.half)
```

### Utilities

```
from sparse.semi_structured import to_sparse_semi_structured
from sparse.decay import masked_add_

# transposable mask select for tensor x
mask_select = TransposableSparse(abs=True)
sparse, mask = mask_select(x)

# fused kernel for grad.add_(p.data * (1 - mask), alpha=decay)
masked_add_(grad, p, mask, alpha=0.03)
```

### Accelerating Transformer Pre-training with 2:4 Sparsity

```
# Contains all of the previous contents, take nanoGPT as an example
# Step 1: remove the original nn.Linear layer
self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)


# Step 2: use 2:4 sparse layer
from sparse.sparse_ops import SparseLinearTranspose
self.c_fc    = SparseLinearTranspose(config.n_embd, 4 * config.n_embd, bias=config.bias)
self.c_proj  = SparseLinearTranspose(4 * config.n_embd, config.n_embd, bias=config.bias)


# Step 3: add masked decay step
optimizer.zero_grad()
for micro_step in range(gradient_accumulation_steps):
    output = model(input)
    loss = loss_fn(output, target)
    scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
######## add masked decay ########
for p in model.parameters():
    if hasattr(p, 'mask') and p.mode == 'sparse':
        masked_add_(p.grad.data, p.data, p.mask, alpha=6e-5)
        p.cnt = 0
######## add masked decay ########
scaler.step(optimizer)
scaler.update()

# Step 4: manually convert to dense fine-tune stage
if iter_num == 250000:
    for p in model.parameters():
        if hasattr(p, 'mask') and p.mode == 'sparse':
            p.mode = 'dense'
```

