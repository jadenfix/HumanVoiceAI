"""
Training script for the HiFi-GAN vocoder.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import random
import time

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from src.vocoder.hifigan import HiFiGAN, DiscriminatorP, MultiPeriodDiscriminator
from src.audio.feature_extractor import FeatureExtractor

class VocoderDataset(Dataset):
    """Dataset for HiFi-GAN vocoder training."""
    
    def __init__(self, data_dir, split='train', sample_rate=22050, segment_length=16384):
        """Initialize the dataset.
        
        Args:
            data_dir: Path to the dataset directory
            split: 'train', 'val', or 'test'
            sample_rate: Audio sample rate
            segment_length: Length of audio segments for training
        """
        self.data_dir = Path(data_dir) / split
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get audio path
        item = self.metadata[idx]
        audio_path = self.data_dir / 'audio' / item['audio_file']
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Randomly crop segment for training
        if waveform.size(-1) >= self.segment_length:
            start = random.randint(0, waveform.size(-1) - self.segment_length)
            waveform = waveform[..., start:start + self.segment_length]
        else:
            # Pad if too short
            padding = self.segment_length - waveform.size(-1)
            waveform = F.pad(waveform, (0, padding), 'constant')
        
        # Extract mel spectrogram
        mel_spectrogram = self.feature_extractor._extract_mel_spectrogram(waveform)
        
        return {
            'waveform': waveform.squeeze(0),  # [time]
            'mel': mel_spectrogram.squeeze(0),  # [n_mels, time]
            'audio_path': str(audio_path)
        }

def feature_loss(fmap_r, fmap_g):
    """Feature matching loss."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss

def discriminator_loss(real_outputs, fake_outputs):
    """Discriminator loss."""
    loss = 0
    for dr, dg in zip(real_outputs, fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss, real_outputs, fake_outputs

def generator_loss(disc_outputs):
    """Generator loss."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses

def train_epoch(generator, mpd, msd, train_loader, 
               g_optimizer, d_optimizer, 
               device, epoch, writer, steps_per_epoch):
    """Train for one epoch."""
    generator.train()
    mpd.train()
    msd.train()
    
    running_d_loss = 0.0
    running_g_loss = 0.0
    running_mel_loss = 0.0
    running_fm_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]', leave=False)
    
    for i, batch in enumerate(progress_bar):
        # Move data to device
        waveform = batch['waveform'].unsqueeze(1).to(device)  # [B, 1, T]
        mel = batch['mel'].to(device)  # [B, n_mels, T']
        
        ##############################
        # (1) Train Discriminators
        ##############################
        d_optimizer.zero_grad()
        
        # Generate audio
        with torch.no_grad():
            fake_audio = generator(mel)
        
        # Real audio
        y = waveform
        y_g_hat = fake_audio
        
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        # Total discriminator loss
        d_loss = loss_disc_s + loss_disc_f
        
        # Backward pass and optimize
        d_loss.backward()
        d_optimizer.step()
        
        ##############################
        # (2) Train Generator
        ##############################
        g_optimizer.zero_grad()
        
        # Generate audio
        y_g_hat = generator(mel)
        
        # MPD
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        # MSD
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        
        # Generator losses
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        
        # Mel-spectrogram loss
        mel_loss = F.l1_loss(mel, generator.feature_extractor._extract_mel_spectrogram(y_g_hat.squeeze(1)))
        
        # Total generator loss
        g_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + 45 * mel_loss
        
        # Backward pass and optimize
        g_loss.backward()
        g_optimizer.step()
        
        # Update running losses
        running_d_loss += d_loss.item()
        running_g_loss += g_loss.item()
        running_mel_loss += mel_loss.item()
        running_fm_loss += (loss_fm_f + loss_fm_s).item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'd_loss': running_d_loss / (i + 1),
            'g_loss': running_g_loss / (i + 1),
            'mel_loss': running_mel_loss / (i + 1)
        })
        
        # Log to TensorBoard
        global_step = epoch * steps_per_epoch + i
        if writer is not None:
            writer.add_scalar('Loss/discriminator', d_loss.item(), global_step)
            writer.add_scalar('Loss/generator', g_loss.item(), global_step)
            writer.add_scalar('Loss/mel', mel_loss.item(), global_step)
            writer.add_scalar('Loss/feature_matching', (loss_fm_f + loss_fm_s).item(), global_step)
    
    # Calculate average losses
    avg_d_loss = running_d_loss / len(train_loader)
    avg_g_loss = running_g_loss / len(train_loader)
    avg_mel_loss = running_mel_loss / len(train_loader)
    avg_fm_loss = running_fm_loss / len(train_loader)
    
    return avg_d_loss, avg_g_loss, avg_mel_loss, avg_fm_loss

def validate(generator, mpd, msd, val_loader, device, epoch, writer):
    """Validate the model."""
    generator.eval()
    mpd.eval()
    msd.eval()
    
    running_d_loss = 0.0
    running_g_loss = 0.0
    running_mel_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1} [Val]', leave=False)
        
        for i, batch in enumerate(progress_bar):
            # Move data to device
            waveform = batch['waveform'].unsqueeze(1).to(device)
            mel = batch['mel'].to(device)
            
            # Generate audio
            y_g_hat = generator(mel)
            
            # Real audio
            y = waveform
            
            # MPD
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            # MSD
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            
            # Generator losses
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            
            # Mel-spectrogram loss
            mel_loss = F.l1_loss(mel, generator.feature_extractor._extract_mel_spectrogram(y_g_hat.squeeze(1)))
            
            # Total generator loss
            g_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + 45 * mel_loss
            
            # Discriminator loss
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            d_loss = loss_disc_s + loss_disc_f
            
            # Update running losses
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            running_mel_loss += mel_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'd_loss': running_d_loss / (i + 1),
                'g_loss': running_g_loss / (i + 1),
                'mel_loss': running_mel_loss / (i + 1)
            })
    
    # Calculate average losses
    avg_d_loss = running_d_loss / len(val_loader)
    avg_g_loss = running_g_loss / len(val_loader)
    avg_mel_loss = running_mel_loss / len(val_loader)
    
    # Log metrics
    if writer is not None:
        writer.add_scalar('Loss/val_discriminator', avg_d_loss, epoch)
        writer.add_scalar('Loss/val_generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/val_mel', avg_mel_loss, epoch)
    
    return avg_d_loss, avg_g_loss, avg_mel_loss

def save_checkpoint(generator, mpd, msd, g_optimizer, d_optimizer, epoch, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'mpd_state_dict': mpd.state_dict(),
        'msd_state_dict': msd.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HiFi-GAN vocoder')
    parser.add_argument('--data_dir', type=str, default='data/vocoder',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vocoder',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/vocoder',
                        help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Sample rate of the audio')
    parser.add_argument('--segment_length', type=int, default=16384,
                        help='Length of audio segments for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize models
    generator = HiFiGAN().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.AdamW(
        generator.parameters(), 
        lr=args.lr, 
        betas=(0.8, 0.99), 
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    d_optimizer = optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=args.lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.999)
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.999)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        mpd.load_state_dict(checkpoint['mpd_state_dict'])
        msd.load_state_dict(checkpoint['msd_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resuming from epoch {start_epoch}")
    
    # Create datasets and dataloaders
    train_dataset = VocoderDataset(
        args.data_dir, 
        split='train',
        sample_rate=args.sample_rate,
        segment_length=args.segment_length
    )
    
    val_dataset = VocoderDataset(
        args.data_dir, 
        split='val',
        sample_rate=args.sample_rate,
        segment_length=args.segment_length
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    print("Starting training...")
    steps_per_epoch = len(train_loader)
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_d_loss, train_g_loss, train_mel_loss, train_fm_loss = train_epoch(
            generator, mpd, msd, train_loader, 
            g_optimizer, d_optimizer, 
            device, epoch, writer, steps_per_epoch
        )
        
        # Validate
        val_d_loss, val_g_loss, val_mel_loss = validate(
            generator, mpd, msd, val_loader, 
            device, epoch, writer
        )
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train D Loss: {train_d_loss:.4f} | "
              f"Train G Loss: {train_g_loss:.4f} | "
              f"Train Mel Loss: {train_mel_loss:.4f} | "
              f"Val D Loss: {val_d_loss:.4f} | "
              f"Val G Loss: {val_g_loss:.4f} | "
              f"Val Mel Loss: {val_mel_loss:.4f}")
        
        # Save checkpoint
        is_best = val_g_loss < best_loss
        if is_best:
            best_loss = val_g_loss
            
        save_checkpoint(
            generator, mpd, msd,
            g_optimizer, d_optimizer,
            epoch, args.checkpoint_dir, is_best
        )
        
        # Log learning rate
        writer.add_scalar('LR/generator', g_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('LR/discriminator', d_scheduler.get_last_lr()[0], epoch)
        
        # Log audio samples
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # Get a validation sample
                sample = next(iter(val_loader))
                sample_mel = sample['mel'].to(device)
                sample_audio = sample['waveform'].unsqueeze(1).to(device)
                
                # Generate audio
                generated_audio = generator(sample_mel)
                
                # Log audio
                writer.add_audio(
                    'Real Audio', 
                    sample_audio[0].cpu(), 
                    epoch, 
                    sample_rate=args.sample_rate
                )
                writer.add_audio(
                    'Generated Audio', 
                    generated_audio[0].cpu(), 
                    epoch, 
                    sample_rate=args.sample_rate
                )
    
    print("Training complete!")
    writer.close()

class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator for HiFi-GAN."""
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(1),
            DiscriminatorP(2),
            DiscriminatorP(4)
        ])
    
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            # Downsample
            if i > 0:
                y = F.avg_pool1d(y, kernel_size=2)
                y_hat = F.avg_pool1d(y_hat, kernel_size=2)
                
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

if __name__ == "__main__":
    main()
