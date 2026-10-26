class OptimizerLR(object):
    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma = gamma
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self, epoch, static_lr):
        self._update_lr(epoch, static_lr)
        self.optimizer.step()
    def _update_lr(self, epoch, static_lr):
        if epoch <= static_lr:
            lr = 0.001
        elif epoch > static_lr:
            lr = 0.001 * (self.gamma ** ((epoch - static_lr) - 1))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr      
    def current_lr(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}\n{'-'*50}")
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    def state_dict(self):
        return self.optimizer.state_dict()

        

class NetworkEval(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        
        self.sample_rate = sample_rate

    @staticmethod
    def _sisnr(reference, estimated, eps: float = 1e-8):
        reference = torch.from_numpy(reference).unsqueeze(0)
        estimate = torch.from_numpy(estimated).unsqueeze(0)
        reference -= reference.mean(dim=-1, keepdim=True)
        estimate -= estimate.mean(dim=-1, keepdim=True)
        projection = (
            (estimate * reference).sum(dim=-1, keepdim=True) * reference
            / (reference.square().sum(dim=-1, keepdim=True) + eps)
        )
        residual = estimate - projection
        return 10 * torch.log10(
            (projection.square().sum(dim=-1, keepdim=True) + eps)
            / (residual.square().sum(dim=-1, keepdim=True) + eps)
        )

    def forward(self, clean, estimate):
        length = estimate.shape[-1]
        clean = clean.detach().cpu().numpy()[:, :length]
        estimate = estimate.detach().cpu().numpy()
        pesq_score = sum(
            pesq_metric(self.sample_rate, ref, est, "wb")
            for ref, est in zip(clean, estimate)
        )
        stoi_score = sum(
            stoi_metric(ref, est, self.sample_rate, extended=False)
            for ref, est in zip(clean, estimate)
        )
        sisnr_score = sum(
            self._sisnr(ref, est)
            for ref, est in zip(clean, estimate)
        )
        
        return pesq_score, stoi_score, sisnr_score