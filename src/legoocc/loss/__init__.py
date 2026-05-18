from mmengine.registry import Registry
GPD_LOSS = Registry('gpd_loss')

from .multi_loss import MultiLoss
from .ce_loss import CELoss, BCELoss
from .plan_reg_loss import PlanRegLoss
from .lovasz_loss import LovaszLoss, GlobalLovaszLoss
from .l2_loss import L2Loss
from .l1_loss import L1Loss
from .sem_geo_loss import Sem_Scal_Loss, Geo_Scal_Loss, Global_Geo_Scal_Loss, Global_Sem_Scal_Loss
from .focal_loss import FocalLoss, GlobalFocalLoss
from .emb_loss import VQVAEEmbedLoss
from .depth_loss import DepthClsLoss, DepthLoss
from .feature_alignment_loss import FeatureAlignmentLoss
