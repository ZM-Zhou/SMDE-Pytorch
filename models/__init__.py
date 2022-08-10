from .get_models import get_losses_with_opts, get_model_with_opts
from .losses.depthhints_loss import DepthHints_PhotoLoss
from .losses.epcdepth_loss import EPCDepth_PhotoLoss
from .losses.md2_photo_loss import MD2_PhotoLoss
from .losses.photo_loss import PhotoLoss
from .losses.seg_loss import SegmentLoss
from.losses.sgt_loss import GuideTripletLoss
from .losses.smooth_loss import SmoothLoss
from .networks.epc_depth import EPCDepth_Net
from .networks.fal_netB import FAL_NetB
from .networks.manydepth import ManyDepth
from .networks.monodepth2 import Monodepth2
from .networks.r_msfm import R_MSFM

__all__ = [
    'get_losses_with_opts', 'get_model_with_opts', 'DepthHints_PhotoLoss',
    'EPCDepth_PhotoLoss', 'GuideTripletLoss','MD2_PhotoLoss', 'PhotoLoss',
    'SegmentLoss','SmoothLoss', 'EPCDepth_Net', 'FAL_NetB', 'ManyDepth',
    'Monodepth2', 'R_MSFM'
]
