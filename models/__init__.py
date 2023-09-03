from .get_models import get_losses_with_opts, get_model_with_opts
from .losses.cost_loss import CostLoss
from .losses.depthhints_loss import DepthHints_PhotoLoss
from .losses.epcdepth_loss import EPCDepth_PhotoLoss
from .losses.grad_loss import GradLoss
from .losses.sgt_loss import GuideTripletLoss
from .losses.hints_loss import HintsLoss
from .losses.md2_photo_loss import MD2_PhotoLoss
from .losses.photo_loss import PhotoLoss
from .losses.seg_loss import SegmentLoss
from .losses.smooth_loss import SmoothLoss
from .networks.epc_depth import EPCDepth_Net
from .networks.fal_netB import FAL_NetB
from .networks.manydepth import ManyDepth
from .networks.monodepth2 import Monodepth2
from .networks.tio_depth import TiO_Depth
from .networks.ocfd_net import OCFD_Net
from .networks.r_msfm import R_MSFM
from .networks.sdfa_net import SDFA_Net


__all__ = [
    'get_losses_with_opts', 'get_model_with_opts', 'CostLoss',
    'DepthHints_PhotoLoss', 'EPCDepth_PhotoLoss', 'GradLoss',
    'GuideTripletLoss','Hintsloss', 'MD2_PhotoLoss', 'PhotoLoss', 'SegmentLoss',
    'SmoothLoss', 'EPCDepth_Net', 'FAL_NetB', 'ManyDepth', 'Monodepth2',
    'TiO_Depth','OCFD_Net', 'R_MSFM', 'SDFA_Net'
]
