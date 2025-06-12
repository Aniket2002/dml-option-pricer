# losses/differential_loss.py

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict

def differential_loss(
    model,
    x: Tensor,
    true_price: Tensor,
    true_delta: Tensor,
    true_vega: Tensor,
    lambda_delta: float = 1.0,
    lambda_vega: float = 1.0
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Returns:
      total_loss = price + λ_delta·delta + λ_vega·vega
      metrics = {'price': price_loss, 'delta': delta_loss, 'vega': vega_loss}
    """
    # ensure we can compute grads
    x = x.clone().detach().requires_grad_(True)

    # predict price
    pred_price = model(x).squeeze(-1)

    # 1) price loss
    price_loss = F.mse_loss(pred_price, true_price)

    # 2) Greeks via AAD
    grads = torch.autograd.grad(
        outputs=pred_price,
        inputs=x,
        grad_outputs=torch.ones_like(pred_price),
        create_graph=True
    )[0]
    pred_delta = grads[:, 0]
    pred_vega  = grads[:, 4]

    # 3) greek losses
    delta_loss = F.mse_loss(pred_delta, true_delta)
    vega_loss  = F.mse_loss(pred_vega,  true_vega)

    # 4) total
    total_loss = price_loss + lambda_delta * delta_loss + lambda_vega * vega_loss

    # detach metrics for logging
    metrics = {
        'price': price_loss.detach(),
        'delta': delta_loss.detach(),
        'vega':  vega_loss.detach()
    }

    return total_loss, metrics
