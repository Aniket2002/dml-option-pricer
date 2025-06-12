# losses/differential_loss.py

import torch
import torch.nn.functional as F

def differential_loss(
    model,
    x: torch.Tensor,
    true_price: torch.Tensor,
    true_delta: torch.Tensor,
    true_vega: torch.Tensor,
    lambda_delta: float = 1.0,
    lambda_vega: float = 1.0
) -> torch.Tensor:
    """
    Instrumented composite loss for DML: logs internal state for debugging.
    """

    print("---- [diff_loss] START ----")
    print(" Input x.requires_grad:", x.requires_grad)
    print(" True price.requires_grad:", true_price.requires_grad)
    print(" True delta.requires_grad:", true_delta.requires_grad)
    print(" True vega.requires_grad:", true_vega.requires_grad)

    # 1) Prepare x for autodiff
    x2 = x.clone().detach()
    print(" After clone().detach(), x2.requires_grad:", x2.requires_grad)
    x2.requires_grad_(True)
    print(" After requires_grad_(True), x2.requires_grad:", x2.requires_grad)

    # 2) Forward pass
    pred_price = model(x2)
    print(" After model(x2):")
    print("  pred_price.requires_grad:", pred_price.requires_grad)
    print("  pred_price.grad_fn:", pred_price.grad_fn)

    # 3) Price loss
    price_loss = F.mse_loss(pred_price, true_price)
    print(" price_loss:", price_loss.item(), "| price_loss.requires_grad:", price_loss.requires_grad)

    # 4) Compute gradients (Greeks) via AAD
    try:
        grads = torch.autograd.grad(
            outputs=pred_price,
            inputs=x2,
            grad_outputs=torch.ones_like(pred_price),
            create_graph=True,
            retain_graph=True
        )[0]
        print(" grads shape:", grads.shape, "| grads.requires_grad:", grads.requires_grad)
    except Exception as e:
        print(" ERROR during autograd.grad:")
        print("  pred_price.requires_grad:", pred_price.requires_grad)
        print("  x2.requires_grad:", x2.requires_grad)
        raise

    # 5) Extract predicted Greeks
    pred_delta = grads[:, 0]
    pred_vega  = grads[:, 4]
    print(" pred_delta.requires_grad:", pred_delta.requires_grad)
    print(" pred_vega.requires_grad:", pred_vega.requires_grad)

    # 6) Greek losses
    delta_loss = F.mse_loss(pred_delta, true_delta)
    vega_loss  = F.mse_loss(pred_vega,  true_vega)
    print(" delta_loss:", delta_loss.item(), "| vega_loss:", vega_loss.item())

    # 7) Composite loss
    loss = price_loss + lambda_delta * delta_loss + lambda_vega * vega_loss
    print(" Composite loss:", loss.item())
    print("---- [diff_loss] END ----\n")

    return loss
