class SENET(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.unfold = nn.Unfold(kernel_size=(1,3), stride=(1, 1), padding=(0, (3-1)//2))
        self.erb = FilterBankERB(65, 64)
        self.stft_extract = RealImagAxes()
        self.enr = FeatEnrich()
        self.dprnn1 = DPGGRU(16, 33, 16)
        self.dprnn2 = DPGGRU(16, 33, 16)
        self.dpcbl1 = DPCNNBlock(16, 16, 2)
        self.dpcbl2 = DPCNNBlock(16, 16, 1)
        self.dpcbl3 = DPCNNBlock(16, 16, 1)
        self.dpcbl4 = DPCNNBlock(16, 16, 2)
        self.decoder = Decoder()
        self.mask = Mask()


    def forward(self, x):
        inp_mask = x  # (B,F,T,2)
        real, imag = self.stft_extract(x)
        real = real.permute(0,2,1)
        imag = imag.permute(0,2,1)
        mag = torch.sqrt(real**2 + imag**2 + 1e-12)
        x = torch.stack([mag, real, imag], dim=1)  
        noisy_resi = self.enr(x[..., :65])
        x = self.erb.compress(x)
        x = self.unfold(x).reshape(x.shape[0], x.shape[1]*3, x.shape[2], x.shape[3])
        
        # Encoder
        x = self.encoder(x)
        
        # DPCNNBlocs & DPGGRUs 
        x = self.dpcbl1(x)
        x = self.dprnn1(x) 
        x = self.dpcbl2(x)
        x = x + noisy_resi
        x = self.dpcbl3(x)
        x = self.dprnn2(x)
        x = self.dpcbl4(x)

        # Decoder and mask
        x = self.decoder(x)
        x = self.erb.decompress(x)

        clean = self.mask(x, inp_mask.permute(0,3,2,1)) # (B,2,T,F)
        clean = clean.permute(0,3,2,1)  # (B,F,T,2)

        return clean