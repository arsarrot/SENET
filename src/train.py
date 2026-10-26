device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SENET().to(device) # define the model
num_epochs = 200 # number of epochs for training
loss_func = NetworkLoss() # define the loss function
eval_metrics = NetworkEval(sample_rate=16000)
length = 48000 # number of samples take in one iteration for training
stride = None # Stride
batch_size = 4 # number of samples in one batch
num_worker = 4 # number of workers
stft_signal = SignalSTFT(n_fft=512, hop_length=256, device=device)

##### Training Dataset
train_dataset = TrainDataset('/kaggle/working/voicebank_wav/train', length=length, stride=stride)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)  

##### Testing Dataset
test_dataset = TestDataset('/kaggle/working/voicebank_wav/test', length=None, stride=None, pad = False)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)   

n_total_steps = len(train_loader) # total number of batches or iterations in one epoch
st_lr = 130 # No learning rate decay till this cap
##### Optimizer
optimizer = OptimizerLR(optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-8), gamma = 0.96)


for epoch in range(num_epochs):
    total_train_loss = 0.0
    model.train()
    for i, (noise, clean, l1, l2) in enumerate(tqdm(train_loader)):
        noise = noise.squeeze(1).to(device)
        clean = clean.squeeze(1).to(device)
        noise = stft_signal.stft(noise)
        true_stft = stft_signal.stft(clean)
        pred_stft = model(noise)
        loss = loss_func(pred_stft, true_stft)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step(epoch+1, st_lr)
        total_train_loss += loss.item()
        if (i + 1) % 400 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    avg_train_loss = total_train_loss / len(train_loader)
    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}. Average Training loss: {avg_train_loss:.4f} ')
    optimizer.current_lr()
    
    total_pesq = 0.0
    total_stoi = 0.0
    total_sisnr = 0.0
    model.eval()
    with torch.no_grad():
        for i, (noise, clean, l1, l2, a) in enumerate(tqdm(test_loader)):
            noise = noise.squeeze(1).to(device)
            clean = clean.squeeze(1).to(device)
            noise  = stft_signal.stft(noise)
            pred_stft = model(noise)
            pred = stft_signal.istft(pred_stft)
            pesq_score, stoi_score, sisnr_score = eval_metrics(clean, pred)
            sisnr_score = sisnr_score.item()
            total_pesq += pesq_score
            total_stoi += stoi_score
            total_sisnr += sisnr_score
        avg_pesq = total_pesq / len(test_loader)
        avg_stoi = total_stoi / len(test_loader)
        avg_sisnr = total_sisnr / len(test_loader)
        print(f'Average PESQ, STOI and SISNR scores are : {avg_pesq:.4f}, {avg_stoi:.4f}, and {avg_sisnr:.4f}')
    checkpoint = {'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth.tar')
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')