_base_ = ['./dal-base.py']

# SPAR-PC: Structure-Prior Aware Refinement for proposal + classification.
# Keep DAL regression branch untouched.
model = dict(
    pts_bbox_head=dict(
        # SPAR-P
        use_spar_p=True,
        spar_p_alpha=0.5,
        # SPAR-C
        use_spar_c=True,
        spar_c_kernel=3,
        spar_c_pool_modes=('max', 'avg'),
        spar_c_ctx_dim=32))
