import torch


def sigreg_strong_loss(embeddings, sketch_dim=64):
    sample_count, feature_count = embeddings.shape
    projection = torch.randn(
        feature_count,
        sketch_dim,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    projection = projection / (projection.norm(p=2, dim=0, keepdim=True) + 1e-6)
    integration_points = torch.linspace(-5, 5, 17, device=embeddings.device)
    gaussian = torch.exp(-0.5 * integration_points.square())
    phases = (embeddings @ projection).unsqueeze(-1) * integration_points
    empirical_real = phases.cos().mean(dim=0)
    empirical_imaginary = phases.sin().mean(dim=0)
    error = (
        (empirical_real - gaussian.unsqueeze(0)).square()
        + empirical_imaginary.square()
    ) * gaussian.unsqueeze(0)
    return torch.trapz(error, integration_points, dim=1).mean() * sample_count

def sigreg_weak_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    """
    N, C = x.size()
    # 1. Sketching (Optional for C=512, but good for consistency)
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 4. Off-diagonal suppression + Diagonal maintenance
    return torch.norm(cov - target, p='fro')