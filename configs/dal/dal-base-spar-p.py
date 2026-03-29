_base_ = ['./dal-base.py']

# SPAR-P only: proposal refinement branch.
model = dict(
    pts_bbox_head=dict(
        # SPAR-P
        use_spar_p=True,
        spar_p_alpha=0.5,
        spar_p_mid_channels=128,
        spar_p_learnable_alpha=True,
        spar_p_use_local_attn=True,
        # SPAR-C
        use_spar_c=False,
        # proposal sampling
        proposal_use_threshold_topk=True,
        proposal_score_threshold=0.05))
