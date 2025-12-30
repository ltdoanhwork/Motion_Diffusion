# DDPMTrainer.py - Fixes & Status Summary

## ✅ Completed Improvements

### 1. **Loss Functions**
- ✅ **MSE Loss (Base)**: Reconstruction loss with masking
- ✅ **Velocity Loss**: Penalizes motion jitter
  - Enabled via `opt.use_velocity_loss`
  - Weight controlled by `opt.velocity_weight`
- ✅ **Acceleration Loss**: Ensures smooth temporal transitions
  - Enabled via `opt.use_acceleration_loss`
  - Weight controlled by `opt.acceleration_weight`
  - Only computes if velocity loss is available
- ✅ **Geometric Loss**: Bone length consistency (DISABLED for now)
  - Enabled via `opt.use_geometric_loss` (but set to `False` in backward_G)
  - Expensive computation, needs more testing

### 2. **Training Optimizations**
- ✅ **EMA (Exponential Moving Average)**: 
  - Decay factor: 0.9999
  - Updated every step via `update_ema()`
  - Separate model for better inference quality
  
- ✅ **Mixed Precision (AMP)**:
  - Uses `autocast()` in forward pass
  - `GradScaler` for backward pass
  - Prevents gradient overflow in FP16
  
- ✅ **Learning Rate Scheduling**:
  - **Warmup**: Linear from 0.1 to 1.0 over 500 steps (or 1 epoch)
  - **Cosine Annealing**: Decays from lr to eta_min=1e-6
  - **Sequential**: Combines both schedulers
  - Step-wise updates (not epoch-wise) for smoother curves

- ✅ **Gradient Clipping**:
  - Norm clipping with norm=0.5 via `clip_grad_norm_`
  - Prevents exploding gradients

### 3. **Code Quality Fixes**
- ✅ **Bug Fix #1**: Velocity/Acceleration loss variable initialization
  - Issue: `vel_gt/vel_pred` undefined when using acceleration loss without velocity loss
  - Fix: Initialize `loss_vel=None` and `loss_acc=None` first, then conditionally define
  - Guard: `if self.opt.use_acceleration_loss and loss_vel is not None:`

## ⚠️ Known Limitations & Future Work

### Geometric Loss
```python
if self.opt.use_geometric_loss and False:  # Disabled for now
```
- Currently disabled because:
  - FK implementation needs validation
  - Expensive to compute (88 bone pairs per batch per timestep)
  - Requires proper axis-angle to position conversion
- To enable: Change `and False` to `and True` after testing

### Forward Kinematics (FK)
- Uses `axis_angle_to_matrix()` from rotation_conversions
- Skeleton tree: 88 parent-child pairs (BEAT skeleton)
- **Assumption**: Input is (B, T, 55, 3) axis-angle format
- **Note**: Model output is (B, T, 264) standardized positions - NOT axis-angle!
  - Need to reshape: `motion.reshape(B, T, 55, 3)` if using FK

## 🔧 Configuration Options (Required in args)

```python
# Loss weights
opt.use_velocity_loss = True        # Enable velocity loss
opt.velocity_weight = 0.1           # Weight for velocity loss
opt.use_acceleration_loss = True    # Enable acceleration loss  
opt.acceleration_weight = 0.05      # Weight for acceleration loss
opt.use_geometric_loss = False      # Keep disabled for now
opt.geometric_weight = 0.01         # Weight if enabled

# Training
opt.lr = 1e-4                       # Learning rate
opt.num_epochs = 100                # Total epochs
opt.batch_size = 32                 # Batch size
opt.log_every = 50                  # Log frequency
opt.save_every_e = 10               # Save checkpoint every N epochs
opt.diffusion_steps = 1000          # Diffusion timesteps
```

## 📊 Expected Training Behavior

### Loss Curves
- **loss_mot_rec**: Should decrease over time (main reconstruction loss)
- **loss_vel**: Decreases as motion becomes smoother
- **loss_acc**: Converges as temporal consistency improves
- **loss_geom**: (If enabled) Bone lengths become consistent

### Learning Rate Schedule
```
Steps 0-500:      Linear warmup from 0.1*lr to lr
Steps 500-end:    Cosine annealing from lr to 1e-6*lr
```

## 🚀 How to Use

```python
# 1. Initialize trainer
from trainers.ddpm_trainer import DDPMTrainer
trainer = DDPMTrainer(args, text_encoder)

# 2. Train
trainer.train(train_dataset)

# 3. Generate (inference)
motion = trainer.generate(text, num_frames=60)

# 4. Use EMA model if desired
trainer.ema_model.eval()
with torch.no_grad():
    motion_ema = trainer.model(...)
```

## ✅ Verification Checklist

Before running training:
- [ ] Verify `opt.batch_size` matches GPU memory
- [ ] Check `opt.diffusion_steps` is reasonable (1000 is standard)
- [ ] Ensure `opt.log_every < len(train_loader)` for logging
- [ ] Validate loss weights sum to reasonable value
- [ ] Test forward pass: `trainer.forward(dummy_batch)` works
- [ ] Test backward pass: `losses = trainer.backward_G()` computes

## 📝 Notes

1. **Velocity/Acceleration Loss**: Only meaningful for temporal motion data (not single frames)
2. **Geometric Loss**: Expensive - only useful for skeleton integrity enforcement
3. **EMA Model**: Improves inference quality but adds memory/computation
4. **AMP**: Speeds up training but may reduce precision (usually OK for this task)
5. **Warmup**: Important for stable training convergence with high learning rates

## 🐛 Potential Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN loss | Exploding gradients | Lower learning rate or increase warmup steps |
| High velocity loss | Jittery motion | Increase `velocity_weight` |
| High geometric loss (if enabled) | Bone distortion | Check FK implementation or skeleton offsets |
| OOM (Out of Memory) | Batch size too large | Reduce `batch_size` or `diffusion_steps` |
| Slow training | EMA updates too frequent | Consider EMA update every N steps instead of every step |

