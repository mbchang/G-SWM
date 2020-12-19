import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from .arch import ARCH
from .module import Flatten, MLP

# VQ
from .vq_modules import VQEmbedding
# VQ


class VQBgModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.embed_size = ARCH.IMG_SIZE // 16
        ###### VQ
        self.enc_output_channels = 512
        self.vq_beta = 1.0
        self.dec_input_channels = 128
        ###### VQ

        # Image encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, self.enc_output_channels, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, self.enc_output_channels),
        
        )
        # self.enc_fc = nn.Linear(self.embed_size ** 2 * self.enc_output_channels, ARCH.IMG_ENC_DIM)

        ###### VQ
        self.enc_fc = nn.Linear(self.enc_output_channels, ARCH.IMG_ENC_DIM)
        self.codebook = VQEmbedding(K=512, D=ARCH.IMG_ENC_DIM)  # dim would correspond to self.embed_size ** 2 * 512
        self.dec_fc = nn.Linear(ARCH.Z_CTX_DIM, self.dec_input_channels)
        ###### VQ

        # self.dec_fc = nn.Linear(ARCH.Z_CTX_DIM, self.embed_size ** 2 * 128)
        # Decoder latent into background
        self.dec = nn.Sequential(
            nn.Conv2d(self.dec_input_channels, 64 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            
            nn.Conv2d(64, 32 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            
            nn.Conv2d(32, 16 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            
            nn.Conv2d(16, 3 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )
        # self.dec = BgDecoder()
        
        self.rnn_post = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.rnn_prior = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.h_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.c_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.h_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.c_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.prior_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
        ###### VQ
        # self.post_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM + ARCH.IMG_ENC_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
        # ****
        self.post_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM + ARCH.IMG_ENC_DIM, 128, 128, ARCH.Z_CTX_DIM], act=nn.CELU())
        ###### VQ

    
    def forward(self, seq):
        return self.encode(seq)
    
    def anneal(self, global_step):
        pass
    
    # def encode(self, seq):
    #     """
    #     Encode input frames into context latents
    #     Args:
    #         seq: (B, T, 3, H, W)

    #     Returns:
    #         things:
    #             bg: (B, T, 3, H, W)
    #             kl: (B, T)
    #     """
    #     # (B, T, 3, H, W)
    #     B, T, C, H, W = seq.size()
        
    #     # Encode images
    #     # (B*T, C, H, W)
    #     enc = self.enc(seq.reshape(B * T, 3, H, W))  # (B*T, C, h, w) --> VQ here
    #     # I deliberately do this ugly thing because for future version we may need enc to do bg interaction
    #     enc = enc.flatten(start_dim=1)  # (B*T, C*h*w)
    #     enc = self.enc_fc(enc)  # (B*T, 128)
    #     enc = enc.view(B, T, ARCH.IMG_ENC_DIM)  # (B, T, 128)
        
    #     h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        
    #     # (B,)
    #     kl_list = []
    #     z_ctx_list = []
    #     for t in range(T):
    #         # Compute posterior
    #         # (B, D)
    #         post_input = torch.cat([h_post, enc[:, t]], dim=-1)
    #         # (B, D), (B, D)
    #         params = self.post_net(post_input)  # VQ
    #         # (B, D), (B, D)
    #         loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
    #         scale = F.softplus(scale) + 1e-4  # VQ
    #         # (B, D)
    #         z_ctx_post = Normal(loc, scale)  # VQ
    #         # (B*T, D)
    #         z_ctx = z_ctx_post.rsample()  # VQ
            
    #         # Compute prior
    #         params = self.prior_net(h_prior)  # VQ
    #         loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
    #         scale = F.softplus(scale) + 1e-4  # VQ
    #         z_ctx_prior = Normal(loc, scale)  # VQ
            
    #         # Temporal encode
    #         h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
    #         h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
    #         # Compute KL Divergence
    #         # (B, D)
    #         kl = kl_divergence(z_ctx_post, z_ctx_prior)  # VQ
    #         assert kl.size()[-1] == ARCH.Z_CTX_DIM
            
    #         # Accumulate things
    #         z_ctx_list.append(z_ctx)
    #         kl_list.append(kl.sum(-1))
        
    #     # (B, T, D) -> (B*T, D)
    #     z_ctx = torch.stack(z_ctx_list, dim=1)
    #     z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
    #     # Before that, let's render our background
    #     # (B*T, 3, H, W)
    #     bg = self.dec(
    #         # z_ctx
    #         self.dec_fc(z_ctx).
    #             view(B * T, 128, self.embed_size, self.embed_size)
    #     )
        
    #     # Reshape
    #     bg = bg.view(B, T, 3, H, W)
    #     z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
    #     # (B, T)
    #     kl_bg = torch.stack(kl_list, dim=1)
    #     assert kl_bg.size() == (B, T)
        
    #     things = dict(
    #         bg=bg,  # (B, T, 3, H, W)
    #         z_ctx=z_ctx,  # (B, T, D)
    #         kl_bg=kl_bg,  # (B, T)  # VQ
    #     )
        
    #     return things


    def encode(self, seq):
        """
        Encode input frames into context latents
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        
        # Encode images
        # (B*T, C, H, W)
        enc = self.enc(seq.reshape(B * T, 3, H, W))  # (B*T, C, h, w) --> VQ here
        # # I deliberately do this ugly thing because for future version we may need enc to do bg interaction


        ###### VQ
        enc = enc.permute((0, 2, 3, 1))  # (B*T, h, w, C)
        enc = enc.reshape(B*T*self.embed_size**2, self.enc_output_channels)  # (B*T*h*w, C)
        enc = self.enc_fc(enc)  # (B*T*h*w, 128)
        enc = enc.reshape(B, T, self.embed_size, self.embed_size, ARCH.IMG_ENC_DIM)  # (B, T, h, w, C)
        enc = enc.permute((0, 2, 3, 1, 4))  # (B, h, w, T, C)
        enc = enc.reshape(B*self.embed_size**2, T, ARCH.IMG_ENC_DIM)

        h_post = self.h_init_post.expand(B*self.embed_size**2, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B*self.embed_size**2, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B*self.embed_size**2, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B*self.embed_size**2, ARCH.RNN_CTX_HIDDEN_DIM)


        # ********
        # enc = enc.flatten(start_dim=1)  # (B*T, C*h*w)
        # enc = self.enc_fc(enc)  # (B*T, 128)
        # enc = enc.view(B, T, ARCH.IMG_ENC_DIM)  # (B, T, 128)
        
        # h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        ###### VQ
        
        # (B,)
        kl_list = []
        z_ctx_list = []
        for t in range(T):
            # Compute posterior

            # (B*h*w, D)
            post_input = torch.cat([h_post, enc[:, t]], dim=-1)
            # (B*h*w, D), (B*h*w, D)
            params = self.post_net(post_input)  # VQ  # (B*h*2, D)

            ###### VQ
            # # (B, D), (B, D)
            # loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
            # scale = F.softplus(scale) + 1e-4  # VQ
            # # (B, D)
            # z_ctx_post = Normal(loc, scale)  # VQ
            # # (B*T, D)
            # z_ctx = z_ctx_post.rsample()  # VQ

            # Compute prior
            # params = self.prior_net(h_prior)  # VQ
            # loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
            # scale = F.softplus(scale) + 1e-4  # VQ
            # z_ctx_prior = Normal(loc, scale)  # VQ
            # *************
            params = params.reshape(B, self.embed_size, self.embed_size, ARCH.Z_CTX_DIM)  # (B, h, w, D)
            params = params.permute((0, 3, 1, 2))  # (B, D, h, w)
            z_ctx_q_x_st, z_ctx_q_x = self.codebook.straight_through(params)  #(B, D, h, w) both 
            ###### VQ
            
            ###### VQ
            # # Temporal encode
            # h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
            # h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            # ************
            # Temporal encode
            z_ctx_q_x_st = z_ctx_q_x_st.permute((0, 2, 3, 1))  # (B, h, w, D)
            z_ctx_q_x_st = z_ctx_q_x_st.reshape(B*self.embed_size**2, ARCH.Z_CTX_DIM)  # (B*h*w, D)

            h_post, c_post = self.rnn_post(z_ctx_q_x_st, (h_post, c_post))  # (B*h*w, D)
            h_prior, c_prior = self.rnn_prior(z_ctx_q_x_st, (h_prior, c_prior))  # (B*h*w, D)
            ###### VQ
            
            ###### VQ
            # # Compute KL Divergence
            # # (B, D)
            # kl = kl_divergence(z_ctx_post, z_ctx_prior)  # VQ
            # assert kl.size()[-1] == ARCH.Z_CTX_DIM
            # ************

            # print('z_ctx_q_x', z_ctx_q_x.shape)
            # print('params', params.shape)

            loss_vq = F.mse_loss(
                z_ctx_q_x.reshape(B, -1), 
                params.detach().reshape(B, -1), 
                reduction='none').mean(-1)
            loss_commit = F.mse_loss(
                params.reshape(B, -1), 
                z_ctx_q_x.detach().reshape(B, -1), 
                reduction='none').mean(-1)
            ###### VQ
            
            ###### VQ
            # # Accumulate things
            # z_ctx_list.append(z_ctx)
            # kl_list.append(kl.sum(-1))
            # ************
            z_ctx_list.append(z_ctx_q_x_st)
            kl_list.append(loss_vq + self.vq_beta*loss_commit)  # probably sum somewhere
            ###### VQ
        
        z_ctx = torch.stack(z_ctx_list, dim=1)  # (B*h*w, T, D)
        dec_z_ctx = self.dec_fc(z_ctx)  # (B*h*w, T, C)
        dec_z_ctx = dec_z_ctx.reshape(B, self.embed_size, self.embed_size, T, self.dec_input_channels)  # (B, h, w, T, C)
        dec_z_ctx = dec_z_ctx.permute((0, 3, 4, 1, 2))  # (B, T, C, h, w)
        dec_z_ctx = dec_z_ctx.reshape(B*T, self.dec_input_channels, self.embed_size, self.embed_size)  # (B*T, C, h, w)

        # z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        # bg = self.dec(
        #     # z_ctx
        #     self.dec_fc(z_ctx).
        #         view(B * T, self.dec_input_channels, self.embed_size, self.embed_size)
        # )
        bg = self.dec(dec_z_ctx)  # (B*T, 3, H, W)

        # print('bg', bg.shape)
        # assert False
        
        # Reshape
        bg = bg.view(B, T, 3, H, W)

        # z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        # # (B, T)
        # kl_bg = torch.stack(kl_list, dim=1)
        # assert kl_bg.size() == (B, T)
        
        # things = dict(
        #     bg=bg,  # (B, T, 3, H, W)
        #     z_ctx=z_ctx,  # (B, T, D)
        #     kl_bg=kl_bg,  # (B, T)  # VQ
        # )

        z_ctx = z_ctx.reshape(B, self.embed_size, self.embed_size, T, ARCH.Z_CTX_DIM)
        z_ctx = z_ctx.permute((0, 3, 4, 1, 2))  # (B, T, C, h, w)
        kl_bg = torch.stack(kl_list, dim=1)  # (B, T)
        assert kl_bg.size() == (B, T)
        
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx,  # (B, T, D)
            kl_bg=kl_bg,  # (B, T)  # VQ
        )
        return things
    
    # def generate(self, seq, cond_steps, sample):
    #     """
    #     Generate new frames given a set of input frames
    #     Args:
    #         seq: (B, T, 3, H, W)

    #     Returns:
    #         things:
    #             bg: (B, T, 3, H, W)
    #             kl: (B, T)
    #     """
    #     # (B, T, 3, H, W)
    #     B, T, C, H, W = seq.size()
        
    #     # Encode images. Only needed for the first few steps
    #     # (B*T, C, H, W)
    #     enc = self.enc(seq[:, :cond_steps].reshape(B * cond_steps, 3, H, W))
    #     # (B*T, D)
    #     enc = enc.flatten(start_dim=1)
    #     # (B*T, D)
    #     enc = self.enc_fc(enc)
    #     # (B, T, D)
    #     enc = enc.view(B, cond_steps, ARCH.IMG_ENC_DIM)
        
    #     h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
    #     # (B,)
    #     z_ctx_list = []
    #     for t in range(T):
            
    #         if t < cond_steps:
    #             # Compute posterior
    #             # (B, D)
    #             post_input = torch.cat([h_post, enc[:, t]], dim=-1)  # VQ
    #             # (B, D), (B, D)
    #             params = self.post_net(post_input)  # VQ
    #             # (B, D), (B, D)
    #             loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
    #             scale = F.softplus(scale) + 1e-4  # VQ
    #             # (B, D)
    #             z_ctx_post = Normal(loc, scale)  # VQ
    #             # (B*T, D)
    #             z_ctx = z_ctx_post.sample()  # VQ
                
    #             # Temporal encode
    #             h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
    #             h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
    #         else:
    #             # Compute prior
    #             params = self.prior_net(h_prior)  # VQ
    #             loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
    #             scale = F.softplus(scale) + 1e-4  # VQ
    #             z_ctx_prior = Normal(loc, scale)  # VQ
    #             z_ctx = z_ctx_prior.sample() if sample else loc  # VQ
    #             h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
    #         # Accumulate things
    #         z_ctx_list.append(z_ctx)
        
    #     # (B, T, D) -> (B*T, D)
    #     z_ctx = torch.stack(z_ctx_list, dim=1)
    #     z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
    #     # Before that, let's render our background
    #     # (B*T, 3, H, W)
    #     bg = self.dec(
    #         # z_ctx
    #         self.dec_fc(z_ctx).
    #             view(B * T, 128, self.embed_size, self.embed_size)
    #     )
    #     bg = bg.view(B, T, 3, H, W)
    #     # Split into lists of length t
    #     z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
    #     things = dict(
    #         bg=bg,  # (B, T, 3, H, W)
    #         z_ctx=z_ctx  # (B, T, D)
    #     )
        
    #     return things

    def generate(self, seq, cond_steps, sample):
        """
        Generate new frames given a set of input frames
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        
        # Encode images. Only needed for the first few steps
        # (B*T, C, H, W)
        enc = self.enc(seq[:, :cond_steps].reshape(B * cond_steps, 3, H, W))
        # (B*T, D)
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = self.enc_fc(enc)
        # (B, T, D)
        enc = enc.view(B, cond_steps, ARCH.IMG_ENC_DIM)
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # (B,)
        z_ctx_list = []
        for t in range(T):
            
            if t < cond_steps:
                # Compute posterior
                # (B, D)
                post_input = torch.cat([h_post, enc[:, t]], dim=-1)  # VQ
                # (B, D), (B, D)
                params = self.post_net(post_input)  # VQ

                ###### VQ
                # # (B, D), (B, D)
                # loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
                # scale = F.softplus(scale) + 1e-4  # VQ
                # # (B, D)
                # z_ctx_post = Normal(loc, scale)  # VQ
                # # (B*T, D)
                # z_ctx = z_ctx_post.sample()  # VQ
                # *************
                z_ctx_q_x_st, z_ctx_q_x = self.codebook.straight_through(params)  #(B, D, h, w) both 
                ###### VQ
                
                ###### VQ
                # # Temporal encode
                # h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
                # h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
                # ************
                # Temporal encode
                h_post, c_post = self.rnn_post(z_ctx_q_x_st, (h_post, c_post))
                h_prior, c_prior = self.rnn_prior(z_ctx_q_x_st, (h_prior, c_prior))
                ###### VQ
            else:
                # Compute prior
                # params = self.prior_net(h_prior)  # VQ
                ###### VQ
                # loc, scale = torch.chunk(params, 2, dim=-1)  # VQ
                # scale = F.softplus(scale) + 1e-4  # VQ
                # z_ctx_prior = Normal(loc, scale)  # VQ
                # z_ctx = z_ctx_prior.sample() if sample else loc  # VQ
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
                # *************
                z_ctx_q_x_st, z_ctx_q_x = self.codebook.straight_through(params)  #(B, D, h, w) both 
                h_prior, c_prior = self.rnn_prior(z_ctx_q_x_st, (h_prior, c_prior))
                ###### VQ
            
            # Accumulate things
            ###### VQ
            # z_ctx_list.append(z_ctx)
            # ******
            z_ctx_list.append(z_ctx_q_x_st)
            ###### VQ
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        bg = bg.view(B, T, 3, H, W)
        # Split into lists of length t
        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx  # (B, T, D)
        )
        
        return things
