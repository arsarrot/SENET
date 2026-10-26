class SignalSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, device):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        window = torch.hann_window(self.n_fft).pow(0.5)
        self.register_buffer('window', window)
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def stft(self, waveform, return_complex = False):
        window = self.window.to(self.device)
        stft_result = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=return_complex,
            pad_mode='reflect'
        )
        return stft_result.to(self.device)
    
    def istft(self, stft_input, return_complex = False):
        window = self.window.to(self.device)
        stft_input = stft_input.to(self.device)
        if stft_input.is_floating_point() and stft_input.shape[-1] == 2:
            stft_input = torch.view_as_complex(stft_input.contiguous())
        waveform = torch.istft(
            stft_input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=False
        )
        return waveform.to(self.device)
    
    def from_real_imag(self, real, imag):
        return real.to(self.device) + 1j * imag.to(self.device)
            
    def reconstruct_from_real_imag(self, real, imag):
        complex_stft = self.from_real_imag(real, imag)
        return self.istft(complex_stft)


class RealImagAxes(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, stft_complex):
        real = stft_complex[:, :, :, 0]
        imag = stft_complex[:, :, :, 1]
        
        return real, imag

class CompressedMSE(nn.Module):
    def __init__(self, compress_factor=0.7, epsilon=1e-12):
        super().__init__()
        
        self.compress_factor = compress_factor
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
        
    def forward(self, est_real, est_imag, tar_real, tar_imag):
        estim_mag = torch.sqrt(est_real**2 + est_imag**2 + self.epsilon)
        tar_mag = torch.sqrt(tar_real**2 + tar_imag**2 + self.epsilon)
        estimate_rc = est_real / (estim_mag ** self.compress_factor)
        estimate_ic = est_imag / (estim_mag ** self.compress_factor)
        target_rc = tar_real / (tar_mag ** self.compress_factor)
        target_ic = tar_imag / (tar_mag ** self.compress_factor)
        real_loss = self.mse_loss(estimate_rc, target_rc)
        imag_loss = self.mse_loss(estimate_ic, target_ic)
        mag_loss = self.mse_loss(estim_mag**(0.3), tar_mag**(0.3))
        total_mse_loss = 30 * (real_loss + imag_loss) + 70 * mag_loss
        
        return total_mse_loss, real_loss, imag_loss, mag_loss


class LossSISNR(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        
        stft_signal = SignalSTFT(n_fft=n_fft, hop_length=hop_length, device=device)
        
    def forward(self, est_real, est_imag, tar_real, tar_imag):
        est_wav = stft_signal.reconstruct_from_real_imag(est_real, est_imag)
        tar_wav = stft_signal.reconstruct_from_real_imag(tar_real, tar_imag)
        projection = (
            (est_wav * tar_wav).sum(dim=-1, keepdim=True) * tar_wav
            / (tar_wav.square().sum(dim=-1, keepdim=True) + 1e-8)
        )
        residual = est_wav - projection
        sisnr_loss = -torch.log10(
            (projection.square().sum(dim=-1, keepdim=True) + 1e-8)
            / (residual.square().sum(dim=-1, keepdim=True) + 1e-8)
        ).mean()
        
        return sisnr_loss, est_wav, tar_wav

class LossPhase(nn.Module):
    def __init__(self):
        super().__init__()
                
    @staticmethod
    def _anti_wrap(x):
        return torch.abs(x - torch.round(x / (2.0 * math.pi)) * 2.0 * math.pi)

    def _phase_losses(self, phase_hat, phase):
        ip_loss = self._anti_wrap(phase_hat - phase).mean()
        gd_loss = self._anti_wrap(torch.diff(phase_hat, dim=1) - torch.diff(phase, dim=1)).mean()
        iaf_loss = self._anti_wrap(torch.diff(phase_hat, dim=2) - torch.diff(phase, dim=2)).mean()
        return ip_loss, gd_loss, iaf_loss

    def forward(self, spec_hat, spec):
        spec_hat = torch.view_as_complex(spec_hat.contiguous())
        spec = torch.view_as_complex(spec.contiguous())
        phase_hat = torch.angle(spec_hat)
        phase = torch.angle(spec)
        ip_loss, gd_loss, iaf_loss = self._phase_losses(phase_hat, phase)
        
        return ip_loss + gd_loss + iaf_loss
        

class NetworkLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stft_extractor = RealImagAxes()
        self.mse_loss = CompressedMSE(compress_factor=0.7, epsilon=1e-12)
        self.sisnr_loss = LossSISNR(n_fft=512, hop_length=256)
        self.phase_loss = LossPhase()
        
    def forward(self, estimate, target):
        device = estimate.device
        phase_loss = self.phase_loss(estimate, target)
        est_real, est_imag = self.stft_extractor(estimate)
        tar_real, tar_imag = self.stft_extractor(target)
        mse_loss, real_loss, imag_loss, mag_loss = self.mse_loss(
            est_real, est_imag, tar_real, tar_imag
        )
        sisnr, est_wav, tar_wav = self.sisnr_loss(
            est_real, est_imag, tar_real, tar_imag
        )
        
        l1_loss = F.l1_loss(est_wav, tar_wav)
        
        total_loss = mse_loss + sisnr + l1_loss + phase_loss
        
        return total_loss