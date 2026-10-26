class FilterBankERB(nn.Module):
    def __init__(self, band_1, band_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        filters = self.filter_bank(band_1, band_2, nfft, high_lim, fs)
        self.register_buffer('erb_fc', filters.T.contiguous())
        self.register_buffer('ierb_fc', filters.contiguous())
        nfreqs = nfft//2 + 1
        self.band_1 = band_1
        self.requires_grad_(False)
        
    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f
    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def filter_bank(self, band_1, band_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = band_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, band_2) # equally spaced erb markers
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32) # converts the erb markers to frequency then mapping to the fequency index of frenquency bins
        # Instantiates an empty tensor of zeros with dimensions [64, 257].
        filters = np.zeros([band_2, nfft // 2 + 1], dtype=np.float32)
        filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(band_2-2):
            filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        filters[-1, bins[-2]:bins[-1]+1] = 1- filters[-2, bins[-2]:bins[-1]+1]

        filters = filters[:, band_1:]
        filters = torch.from_numpy(np.abs(filters))
        return filters
        
    def compress(self, x):
        x_low = x[..., :self.band_1].to(device)
        x_high = torch.matmul(x[..., self.band_1:], self.erb_fc)
        return torch.cat([x_low, x_high], dim=-1)
    def decompress(self, x):
        x_erb_low = x[..., :self.band_1]
        x_erb_high = torch.matmul(x[..., self.band_1:], self.ierb_fc)
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


        
class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        batch, dual_c, time, freq = x.shape
        channel = dual_c // 2
        x = x.view(batch, 2, channel, time, freq)
        x = x.transpose(1, 2)
        
        return x.reshape(batch, dual_c, time, freq)
        
        
class ConvBNPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, decode=False, tanH=False):
        super().__init__()
        
        conv_op = nn.ConvTranspose2d if decode else nn.Conv2d
        self.conv = conv_op(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if tanH else nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups,  decode = False):
        super().__init__()
        
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.linear = nn.Linear(self.in_channels, self.out_channels * 2)
        self.glu = nn.GLU()
        conv_op = nn.ConvTranspose2d if decode else nn.Conv2d
        self.dtcl = conv_op(self.in_channels, self.out_channels, kernel_size, stride, padding, groups = self.in_channels)
        self.shuffle = Shuffle()  
        
    def forward(self, x): 
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.norm(x1.permute(0, 1, 3, 2)).permute(0, 3, 2, 1)
        x1 = self.linear(x1)
        x1 = self.glu(x1).permute(0, 3, 1, 2)
        x1_resi = self.dtcl(x1)
        x1 = x1 + x1_resi
        x = self.shuffle(x1, x2)
        
        return x                

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_enc = nn.Sequential(
            ConvBNPRelu(in_channels = 9, out_channels = 16, kernel_size = (1, 5), stride = (1, 2), padding = (0, 2), groups = 1, decode=False),
            ConvBNPRelu(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 2), padding = (0, 2), groups = 2, decode=False),
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = False),
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = False),
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = False)
        )
        
    def forward(self, x):
        x = self.conv_enc(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_dec = nn.Sequential(
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = True),
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = True),
            ConvNet(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 1), padding = (0, 2), groups = 8, decode = True),
            ConvBNPRelu(in_channels = 16, out_channels = 16, kernel_size = (1, 5), stride = (1, 2), padding = (0, 2), groups = 2, decode=True),
            ConvBNPRelu(in_channels = 16, out_channels = 2, kernel_size = (1, 5), stride = (1, 2), padding = (0, 2), groups = 1, decode=True, tanH=True)
        )
                
    def forward(self, x):
        x = self.conv_dec(x)
        
        return x             

class ConvMod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.norm = nn.BatchNorm2d(in_channels)
        self.linear = nn.Linear(in_channels, out_channels * 2)
        self.glu = nn.GLU()
        self.dwc1d = nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel_size, stride=(1,1), groups = in_channels, padding=padding)
        self.shuffle = Shuffle()    
        
    def forward(self, x): 
        x = self.norm(x).permute(0, 2, 3, 1)
        x = self.linear(x)
        x = self.glu(x).permute(0, 3, 1, 2)
        x_resi = self.dwc1d(F.pad(x, (0, 0, self.kernel_size[0]-1, 0)))
        x = x + x_resi
        
        return self.shuffle(x)   


class OptimizableGate(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        x = self.beta * torch.sigmoid(self.slope * x)
        return x


class DPCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(DPCNN, self).__init__()
        
        self.inter = nn.ModuleList(ConvMod(in_channels//2, out_channels//2, (3, 1), (0, 0)) for _ in range(2))
        self.intra = nn.ModuleList(ConvMod(in_channels//2, out_channels//2, (1, 3), (0, 1)) for _ in range(3))
        
        self.optgate = OptimizableGate(in_channels//2)
        self.shuffle = Shuffle()
    
    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        res_ter = x1
        res_tra = x2
        # Inter_Conv
        x1 = self.inter[0](x1)
        x1 = x1.mean(3, keepdim=True) 
        x1 = self.inter[1](x1) + res_ter
        # Intra_Conv       
        x = self.intra[0](x2)
        x2 = x.mean(3, keepdim=True) 
        x2 = self.intra[1](self.optgate(x) * x2)
        x2 = self.intra[2](self.optgate(x) * x2)
        x = self.shuffle(x1, (self.optgate(x) * x2) + res_tra)
        
        return x


class DPCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()

        self.layers = nn.ModuleList([
            DPCNN(in_channels, out_channels) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  
            
        return x 


class FeatEnrich(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.local1 = ConvBNPRelu(3, 16, (1,5), (1,2), (0,2), 1, decode=False)
        self.local2 = DPCNNBlock(16, 16, 2)
        
    def forward(self, x):
        x = self.local1(x)
        x = self.local2(x)
        
        return x             
        

class GroupedGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers = 1, batch_first = True, bidirectional = False):
        super().__init__()
        self.directions = 2 if bidirectional else 1
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.gru = nn.ModuleList(
            nn.GRU(input_size = in_dim // 2, 
                   hidden_size = hid_dim // 2, 
                   num_layers = n_layers, batch_first = batch_first, 
                   bidirectional = bidirectional) for _ in range(2))

    def forward(self, x, h = None):
        x1, x2 = x.chunk(2, dim=-1)
        hid_shape = (
        self.n_layers * self.directions,
            x.shape[0],
            self.hid_dim,
        )
        if h is None:
            h = x.new_zeros(hid_shape)
        h1, h2 = h.chunk(2, dim=-1)
        y1, h1 = self.gru[0](x1, h1.contiguous())
        y2, h2 = self.gru[1](x2, h2.contiguous())
        return torch.cat((y1, y2), dim=-1)


class DPGGRU(nn.Module):
    def __init__(self, in_dim, freq, hid_dim, **kwargs):
        super().__init__()
        self.intra = nn.Sequential(
            GroupedGRU(in_dim=in_dim, hid_dim=hid_dim // 2, bidirectional=True),
            nn.Linear(hid_dim, hid_dim))
        self.ln_tra = nn.LayerNorm((freq, hid_dim), eps=1e-8)

        self.inter = nn.Sequential(
            GroupedGRU(in_dim=in_dim, hid_dim=hid_dim, bidirectional=False),
            nn.Linear(hid_dim, hid_dim))
        self.ln_ter = nn.LayerNorm((freq, hid_dim), eps=1e-8)
        self.shuffle = Shuffle()

    def forward(self, x):
        b, _, t, f = x.shape
        x = self.shuffle(x).permute(0, 2, 3, 1)  
        tra_out = x + (self.ln_tra(self.intra(x.reshape(b * t, f, -1)).reshape(b, t, f, -1)))
        inter = self.ln_ter(
            self.inter(tra_out.permute(0, 2, 1, 3).reshape(b * f, t, -1)).reshape(b, f, t, -1).permute(0, 2, 1, 3)
        )
        return self.shuffle((tra_out + inter).permute(0, 3, 1, 2))


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mask, noisy):
        s_real = noisy[:,0] * mask[:,0] - noisy[:,1] * mask[:,1]
        s_imag = noisy[:,1] * mask[:,0] + noisy[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1) 
        
        return s
