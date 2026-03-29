_base_ = ['./dal-base.py']

# SPAR-C only: classification refinement branch.
model = dict(
    pts_bbox_head=dict(
        # SPAR-P
        use_spar_p=False,
        # SPAR-C
        use_spar_c=True,
        spar_c_kernel=3,
        spar_c_pool_modes=('max', 'avg'),
        spar_c_ctx_dim=32,
        spar_c_mlp_layers=2,
        # proposal sampling (baseline top-K to isolate C-branch effect)
        proposal_use_threshold_topk=False,
        proposal_score_threshold=0.0))
