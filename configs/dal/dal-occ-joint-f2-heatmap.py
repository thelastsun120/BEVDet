_base_ = ['./dal-occ-joint-base.py']

model = dict(
    pts_bbox_head=dict(
        occ_feedback='heatmap',
    ))

