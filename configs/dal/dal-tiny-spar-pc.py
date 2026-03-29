_base_ = ['./dal-tiny.py']

# SPAR-PC on DAL-Tiny.
model = dict(
    pts_bbox_head=dict(
        # SPAR-P
        use_spar_p=True,
        spar_p_alpha=0.5,
        spar_p_mid_channels=128,
        spar_p_learnable_alpha=True,
        spar_p_use_local_attn=True,
        # SPAR-C
        use_spar_c=True,
        spar_c_kernel=3,
        spar_c_pool_modes=('max', 'avg'),
        spar_c_ctx_dim=32,
        spar_c_mlp_layers=2,
        # proposal sampling
        proposal_use_threshold_topk=True,
        proposal_score_threshold=0.05))
