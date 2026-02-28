from options.base_options import BaseOptions
import argparse

class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # ==================== Model Architecture ====================
        self.parser.add_argument('--num_layers', type=int, default=8, help='num_layers of transformer')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim of transformer')
        self.parser.add_argument('--no_clip', action='store_true', help='whether use clip pretrain')
        self.parser.add_argument('--no_eff', action='store_true', help='whether use efficient attention')

        # ==================== Diffusion Settings ====================
        self.parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion_steps of transformer')
        self.parser.add_argument('--beta_schedule', type=str, default='linear',
                                 choices=['linear', 'cosine', 'scaled_linear'],
                                 help='Beta schedule for diffusion process')
        self.parser.add_argument('--prediction_type', type=str, default='epsilon',
                                 choices=['epsilon', 'v_prediction', 'x_start'],
                                 help='What the model predicts (noise, v, or x0)')
        
        # ==================== Training Hyperparameters ====================
        self.parser.add_argument('--num_epochs', type=int, default=51, help='Number of epochs')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
        self.parser.add_argument('--times', type=int, default=1, help='times of dataset')
        self.parser.add_argument('--feat_bias', type=float, default=25, help='Scales for global motion features and foot contact')
        self.parser.add_argument('--test_ratio', type=float, default=0.1,
                                 help='Split ratio for test.txt when auto-creating BEAT splits')
        self.parser.add_argument('--split_seed', type=int, default=3407,
                                 help='Random seed used for train/test split generation')
        
        # ==================== Motion-Specific Losses ====================
        self.parser.add_argument('--use_velocity_loss', action='store_true',
                                 help='Use velocity matching loss for temporal smoothness')
        self.parser.add_argument('--use_acceleration_loss', action='store_true',
                                 help='Use acceleration matching loss for motion dynamics')
        self.parser.add_argument('--use_geometric_loss', action='store_true',
                                 help='Use geometric consistency loss (bone length)')
        self.parser.add_argument('--use_fk_loss', action='store_true',
                                 help='Use forward kinematics (FK) loss for global position accuracy')
        self.parser.add_argument('--velocity_weight', type=float, default=0.5,
                                 help='Weight for velocity loss')
        self.parser.add_argument('--acceleration_weight', type=float, default=0.1,
                                 help='Weight for acceleration loss')
        self.parser.add_argument('--geometric_weight', type=float, default=0.3,
                                 help='Weight for geometric loss')
        self.parser.add_argument('--fk_weight', type=float, default=1.0,
                                 help='Weight for forward kinematics (FK) loss')

        # ==================== Training Improvements ====================
        self.parser.add_argument('--ema_decay', type=float, default=0.9999,
                                 help='EMA decay rate for model weights (0 to disable)')
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                 help='Number of gradient accumulation steps')
        self.parser.add_argument('--use_amp', action='store_true',
                                 help='Use automatic mixed precision training')
        self.parser.add_argument('--max_grad_norm', type=float, default=1.0,
                                 help='Maximum gradient norm for clipping')
        
        # ==================== Classifier-Free Guidance ====================
        self.parser.add_argument('--cfg_dropout', type=float, default=0.1,
                                 help='Probability to drop text condition for CFG training')
        self.parser.add_argument('--cfg_scale', type=float, default=4.5,
                                 help='CFG scale for inference (1.0 = no guidance)')

        # ==================== Checkpointing & Logging ====================
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=10, help='Frequency of saving models (by epoch)')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving models (by iteration)')
        
        self.is_train = True
