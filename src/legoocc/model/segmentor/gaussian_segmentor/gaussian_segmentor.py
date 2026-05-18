import torch
import shutil
import numpy as np
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
import sys

from depth_anything_v2.dpt import DepthAnythingV2
# from legoocc.model.depthbranch.depthnet import DepthNet
# from legoocc.model.depthbranch.unet2d import DecoderBN
import torch.nn as nn
from PIL import Image
import cv2
import torch.nn.functional as F
import os
# import clip
import numpy as np
import copy
import matplotlib.pyplot as plt
import open3d as o3d
from omegaconf import OmegaConf
from einops import einsum, repeat, rearrange
from peft import get_peft_model, LoraConfig, TaskType
from torchvision import transforms
from torch.utils import dlpack
from torch_cluster import radius
from torch_scatter import scatter_add, scatter_max, scatter_min

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from trident import Trident, preprocess_image
from segment_anything import SamAutomaticMaskGenerator

from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl,
    apply_activation_checkpointing, CheckpointWrapper
)

from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix, safe_get_quaternion
from ...encoder.gaussianformer.gaussian_encoder_layer import SparseGaussian3DEncoder
from .imagenet_template import openai_imagenet_template, sub_imagenet_template

from .utils import set_input_size
from legoocc.model.head.gaussian_occ_head.gsplat_rasterization import rasterize_gaussians


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def inverse_sigmoid(x):
    return torch.log(x/((1-x)+1e-10))


@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        flag_depthbranch=False,
        flag_depthanything_as_gt=False,
        depthbranch=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None, 
        init_cfg=None,

        #
        with_ov=False,
        ov_align_loss=False,

        # 
        test_lang_feat=None,

        load_ov_model=True,

        **kwargs,
    ):
        super().__init__(init_cfg)

        self.with_ov = with_ov
        self.ov_align_loss = ov_align_loss
        self._text_embed_cache = {}

        if test_lang_feat is not None:
            self.register_buffer('_test_lang_feat', torch.load(test_lang_feat)['embeddings'])
        else:
            self._test_lang_feat = None

        # hard code
        if with_ov:
            self.semantic_dim = 0
        else:
            self.semantic_dim = 13

        self.flag_depthbranch = flag_depthbranch
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                # depth branch
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
                checkpoint = torch.load(f"{os.getenv('HF_HOME', os.path.expanduser('~') + '/.cache/huggingface')}/hub/EmbodiedOcc/finetune_scannet_depthanythingv2.pth", map_location='cpu')['model']
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_key = k[len('module.'):] 
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                self.depthanything.load_state_dict(new_state_dict)

            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()

            self.backbone = basemodel

            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )

        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if future_decoder is not None: 
            self.future_decoder = MODELS.build(future_decoder)
        if head is not None:
            self.head = MODELS.build(head)

        if load_ov_model:

            # build for ov
            sam_checkpoint = os.path.join(
                os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
                "hub",
                "checkpoints",
                "sam_vit_l_0b3195.pth"
            )
            model_type = "vit_l"
            self.ov_model = Trident(
                clip_type='openai',
                model_type='ViT-B/16',
                vfm_model='dino',
                name_path='./config/my_name.txt',
                sam_refinement=True,
                coarse_thresh=0.2,
                minimal_area=225,
                debug=True,
                sam_ckpt=sam_checkpoint,
                sam_model_type=model_type)
            for n, p in self.ov_model.named_parameters():
                p.requires_grad = False
            self.ov_model.eval()
        self.ov_feat_dim = 512

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep Trident frozen in inference mode even when GaussianSegmentor trains.
        if hasattr(self, 'ov_model') and self.ov_model is not None:
            self.ov_model.eval()
        return self

    # @torch.no_grad()
    # def get_word_embed(self, word: str, device: torch.device = None) -> torch.Tensor:
    #     device = device or self.device

    #     key = word.strip().lower()
    #     if key in self._text_embed_cache:
    #         return self._text_embed_cache[key].to(device, non_blocking=True)

    #     prompts = [tpl(key) for tpl in openai_imagenet_template]

    #     query = self.ov_model.tokenizer([temp(word) for temp in openai_imagenet_template]).to(device)
    #     feature = self.ov_model.clip.encode_text(query).float()
    #     feature /= feature.norm(dim=-1, keepdim=True).clamp(min=1e-2)
    #     feature = feature.mean(dim=0)

    #     # feature = feature / feature.norm().clamp(1e-2)

    #     feat_cpu = feature.detach().to('cpu')
    #     self._text_embed_cache[key] = feat_cpu
    #     return feature

    @torch.no_grad()
    def get_fm_feat(seg, self, image, metas):
        
        def get_presave_feat_path(img_path):
            path_parts = img_path.split('/')
            target_dir = os.path.join(*path_parts[:-3], 'trident_feat')
            target_path = os.path.join(target_dir, path_parts[-2], path_parts[-1] + '.pth')
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            return target_path

        # copy from trident.py

        stride = self.clip_stride
        all_feats = []

        all_image = rearrange(image, 'b f c h w -> (b f) c h w')
        for batch_idx, single_img in enumerate(all_image):

            out_feats = []

            # img_path = metas[batch_idx]['rgb_path']
            all_images = [metas[batch_idx]['rgb_path']] + (metas[batch_idx]['neighbor_frames'] if (seg.training and 'neighbor_frames' in metas[batch_idx]) else [])
            # all_images = [metas[batch_idx]['rgb_path']]
            for img_path in all_images:

                clip_features = None
                presave_feat_path = get_presave_feat_path(img_path)

                if os.path.exists(presave_feat_path):
                    try:
                        clip_features = torch.load(presave_feat_path).cuda()
                    except:
                        clip_features = None

                if clip_features is None:
                    # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=single_img.device, dtype=single_img.dtype).view(-1, 1, 1)
                    # std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=single_img.device, dtype=single_img.dtype).view(-1, 1, 1)

                    # src_img = (single_img - mean) / std
                    # src_img = src_img[None] # this need a batch dim

                    img = Image.open(img_path)
                    img_tensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                    ])(img)
                    src_img = img_tensor.unsqueeze(0).to('cuda')

                    # tmp_img = cv2.imread(img_path)
                    # tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                    tmp_img = np.array(img)   # [H,W,C], RGB
                    tmp_h, tmp_w = tmp_img.shape[:2]
                    if tmp_h % stride != 0: tmp_h = (tmp_h // stride + 1) * stride
                    if tmp_w % stride != 0: tmp_w = (tmp_w // stride + 1) * stride
                    tmp_img = cv2.resize(tmp_img, (tmp_w, tmp_h))

                    sam_enc_feats, sam_attn, sam_v, sam_valid_h, sam_valid_w = self.get_sam_feat(tmp_img, 16)

                    processed_img = preprocess_image(src_img, stride, self.slide_crop)
                    clip_whole_h, clip_whole_w = processed_img.shape[-2:]
                    clip_feat_h, clip_feat_w = clip_whole_h // stride, clip_whole_w // stride
                    img_batch, paddings, patch_locs, win_sizes = self.get_windowed_imgs(processed_img, stride)

                    imgs_norm = [self.norm(self.unnorm(img_batch[i])) for i in range(len(img_batch))]  # replace norm here
                    imgs_norm = torch.stack(imgs_norm, dim=0)
                    imgs_norm = imgs_norm.half()
                    feat_out = {}

                    def hook_fn_forward_qkv(module, input, output):
                        feat_out["qkv"] = output

                    if self.vfm_model == 'dino':
                        self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                    # Forward pass in the model
                    patch_size = self.vfm.patch_embed.patch_size
                    if type(patch_size) is tuple: patch_size = patch_size[0]
                    feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
                    nb_im = feat.shape[0]  # Batch size
                    vfm_h, vfm_w = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size
                    vfm_feats = feat[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1).permute(0, 3, 1, 2) #batch, c, h, w

                    clip_features = self.clip.encode_image(
                        img_batch.half(),
                        external_feats=vfm_feats, beta=self.beta, gamma=self.gamma,
                        paddings=paddings,dst_coords=patch_locs,win_sizes=win_sizes,
                        dst_vh=clip_feat_h, dst_vw=clip_feat_w, sam_attn=sam_attn, sam_v=sam_v,
                        cos_fac=self.cos_fac, vfm_token_size = (vfm_h, vfm_w),
                        refine_neg_cos=self.refine_neg_cos)

                    clip_features = rearrange(clip_features, 'b (h w) c -> b c h w', h=sam_valid_h, w=sam_valid_w)
                    torch.save(clip_features, presave_feat_path)

                out_feats.append(clip_features)

            all_feats.append(torch.stack(out_feats).squeeze(1))
            # .view(*image.shape[:2], -1, *image.shape[-2:])
        return torch.stack(all_feats).unflatten(0, image.shape[:2])

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640
        
        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)
            
        img_feats_backbone = feature_x

        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone) # dict

        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped, img_feats_out['1_1'] # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

    def obtain_bev(self, imgs, metas):
        B, f, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*f, N, C, H, W)

        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs) # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

        if self.flag_depthbranch:
            if self.flag_depthanything_as_gt:
                # depth branch
                self.depthanything.eval()
                image_ = metas[0]['img_depthbranch']
                depth_pred = self.depthanything.infer_image(image_, 480, 640, 480)
                depthnet_output = depth_pred
            else:  
                depthnet_output = None
        else:
            depthnet_output = None

        anchor, instance_feature, depth2occ, depthnet_output_loss, predtoreturn = self.lifter(self.flag_depthbranch, self.flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas)    # b, g, c 

        anchor, feats = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # b, g, c
        return anchor, depth2occ, depthnet_output_loss, predtoreturn, feats

    def test_lang_feat(self, test_prompt=None):
        if test_prompt is not None:
            query_features = []
            for qw in test_prompt:
                query = self.ov_model.tokenizer([temp(qw) for temp in openai_imagenet_template]).cuda()
                # query = self.ov_model.tokenizer([test_prompt]).cuda()
                feature = self.ov_model.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
            return torch.cat(query_features, dim=0).detach().float()

        if self._test_lang_feat is None:
            lang_feat = self.ov_model.query_features.float()
            # furn_emb = remove_subspace(lang_feat[9], lang_feat[:9])
            # objs_emb = remove_subspace(lang_feat[10], lang_feat[:9])
            # lang_feat[9] = furn_emb
            # lang_feat[10] = objs_emb
            return lang_feat
        return self._test_lang_feat

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        test_prompt=None,
        **kwargs,
    ):

        B, f, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < f
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
            feats = None
        else:
            bev, depth2occ, depthnet_output_loss, predtoreturn, feats = self.obtain_bev(imgs, metas)

        feats = feats[-1] if isinstance(feats, list) else feats

        # BF, H, W, C = bev.shape
        BF, G, C = bev.shape # bev is actually anchors [1, 21600, 24]
        bev = bev.reshape(B, f, G, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()

        if self.with_ov or self.ov_align_loss:
            # get foundation model feat
            with torch.no_grad():
                fm_feats = self.get_fm_feat(self.ov_model, rearrange(imgs, 'b f v c h w -> b (f v) c h w'), metas)
        else:
            fm_feats = None

        output_dict = self.head(
            bev_feat=bev_predict,  # [1, 1, 21600, 24]
            points=points,
            label=label,
            output_dict=output_dict, 
            metas=metas,
            inst_feats=feats,
            test_mode=test_mode,
            fm_feats=fm_feats)

        return output_dict, depth2occ, predtoreturn

    def frame_split(self, grad_frames, imgs, metas):
        f = imgs.shape[1]
        index = np.random.permutation(f)
        inv_index = np.argsort(index)
        imgs_grad = imgs[:, index[:grad_frames]]
        imgs_no_grad = imgs[:, index[grad_frames:]]
        metas_grad = deepcopy(metas)
        metas_no_grad = deepcopy(metas)
        for meta, meta_grad, meta_no_grad in zip(metas, metas_grad, metas_no_grad):
            lidar2img = np.asarray(meta['lidar2img'])
            meta_grad['lidar2img'] = lidar2img[index[:grad_frames]]
            meta_no_grad['lidar2img'] = lidar2img[index[grad_frames:]]
            img_aug_matrix = meta['img_aug_matrix']
            meta_grad['img_aug_matrix'] = img_aug_matrix[index[:grad_frames]]
            meta_no_grad['img_aug_matrix'] = img_aug_matrix[index[grad_frames:]]

        return imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index
    
    def forward_autoreg(self,
                        imgs=None,
                        metas=None,
                        points=None,
                        label=None,
                        test_mode=True,
                        **kwargs,
        ):
        B, f, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'

        bev = self.obtain_bev(imgs, metas)
        BF, G, C = bev.shape # bev is actually anchors
        bev = bev.reshape(B, f, G, C)

        output_dict = self.future_decoder.forward_autoreg(bev, metas)
        bev_predict = output_dict.pop('bev')
        output_dict = self.head(
            bev_feat=bev_predict, 
            points=points, 
            label=label, 
            output_dict=output_dict, 
            metas=metas,
            test_mode=test_mode)

        return output_dict



@MODELS.register_module()
class VGGTGaussianSegmentor(GaussianSegmentor):

    def __init__(
        self,
        *args,
        pretrained_path=None,
        text_prompts=None,
        frozen_backbone=True,
        freeze_blocks=-1,
        # lora_config=None,
        semantic_dim=13,
        scale_range=[0.01, 0.08],
        include_opa=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        num_bins=10,
        with_unc=False,
        with_opacity=True,
        extra_sparse_gaussian=None,
        opacities_threshold=0.,
        densities_threshold=None,
        use_depthanything=False,
        use_dino=False,
        dino_backbone='dinov2_vitb14',

        bin_logit_scale=1.0,
        min_max_temperature=None,

        # bkb params
        target_size=420,
        patch_size=14,

        with_cp=False,

        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def _module_require_grad(m, flag=False):
            for n, p in m.named_parameters():
                p.requires_grad = flag

        self.with_cp = with_cp
        self.target_size = target_size
        self.patch_size = patch_size
        self._text_embed_cache = {}
        # self.cos_sim_scale = cos_sim_scale
        self.bin_logit_scale = bin_logit_scale
        if min_max_temperature:
            assert len(min_max_temperature) == 2 and min(min_max_temperature) > 0
            assert min_max_temperature[0] < min_max_temperature[1]
        self.min_max_temperature = min_max_temperature

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_depthanything = use_depthanything

        self.use_dino = use_dino
        if use_depthanything:
            # depth branch
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
            # checkpoint = torch.load('/data1/code/wyq/gaussianindoor/EmbodiedOcc/checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
            checkpoint = torch.load(f"{os.getenv('HF_HOME', os.path.expanduser('~') + '/.cache/huggingface')}/hub/EmbodiedOcc/finetune_scannet_depthanythingv2.pth", map_location='cpu')['model']
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_key = k[len('module.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            depthanything.load_state_dict(new_state_dict)
            self.backbone = depthanything

            _module_require_grad(self.backbone, True)

            if with_cp:
                blk_type = type(self.backbone.pretrained.blocks[0])
                def check_fn(m: nn.Module) -> bool:
                    return isinstance(m, blk_type)
                wrap_fn = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,  # 推荐
                    preserve_rng_state=True,                      # 保持dropout/droppath一致
                )
                apply_activation_checkpointing(self.backbone.pretrained, checkpoint_wrapper_fn=wrap_fn, check_fn=check_fn)
                vit = self.backbone.pretrained
                num_wrapped = sum(1 for m in vit.modules() if isinstance(m, CheckpointWrapper))
                print("Gradient Checkpoint Wrapped Blocks:", num_wrapped)

            self.gs_head = copy.deepcopy(self.backbone.depth_head)
            for n, p in self.gs_head.named_parameters():
                p.requires_grad = True
            self.backbone.depth_head = None
            self.backbone.pretrained.mask_token.requires_grad = False
            self.gs_head.scratch.refinenet4.resConfUnit1 = None
            self.gs_head.scratch.output_conv2 = None
            self.bkb_feat_dim = self.backbone.pretrained.embed_dim

        else:
            assert False

        # Build text prototype embeddings
        if text_prompts is not None:
            text_proto_embeds = build_text_proto_embeds(self, text_prompts)
            self.register_buffer('text_proto_embeds', text_proto_embeds)
        else:
            self.text_proto_embeds = None

        self.frozen_backbone = frozen_backbone

        self.with_unc = with_unc
        self.with_opacity = with_opacity

        # gs params
        self.scale_range = scale_range
        self.include_opa = include_opa
        self.semantic_dim = semantic_dim
        self.semantic_start = 10 + int(include_opa)

        self.cuda_kwargs = cuda_kwargs
        from legoocc.model.head.gaussian_occ_head.ops.localagg_prob.local_aggregate_prob import LocalAggregator
        self.aggregator = LocalAggregator(**cuda_kwargs)

        _dim_ = 256
        self.gs_pred_layer = nn.Sequential(
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_,
                      semantic_dim + 3 + 4 + (1 if with_opacity else 0) + (1 if with_unc else 0))
        )
        self.anchor_feat_layer = nn.Sequential(
            nn.Conv2d(64 if self.use_depthanything else 128, _dim_, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
        )
        self.anchor_depth_pred = nn.Sequential(
            nn.Conv2d(_dim_, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.num_bins = num_bins
        if num_bins > 0:
            self.bins_emb = nn.Embedding(num_bins, _dim_)
            self.depth_scale_layer = nn.Conv2d(_dim_, 1, kernel_size=1)

        self.opacities_threshold = opacities_threshold
        self.densities_threshold = densities_threshold

        if densities_threshold is not None:
            assert opacities_threshold is None
        if opacities_threshold is not None:
            assert densities_threshold is None

        self.extra_sparse_gaussian = extra_sparse_gaussian
        if extra_sparse_gaussian:
            self.sparse_gs_pred_layer = nn.Sequential(
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_,
                        semantic_dim + 3 + 4 + (1 if with_unc else 0))
            )
            self.sparse_anchor_feat_layer = nn.Sequential(
                nn.Conv2d(128, _dim_, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
            )
            self.sparse_bins_emb = nn.Embedding(num_bins, _dim_)

        self.feat2query = nn.Sequential(
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_, self.ov_feat_dim))
        self.xyz2qpos = nn.Linear(3, self.ov_feat_dim)

        self.kv_proj = nn.Sequential(
            nn.Linear(self.bkb_feat_dim, self.ov_feat_dim),
            nn.ReLU(),
            nn.Linear(self.ov_feat_dim, self.ov_feat_dim))
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=self.ov_feat_dim, num_heads=4,
            num_points=4, num_levels=4,
            batch_first=True, dropout=0.1)
        self.deform_attn.init_weights()

        self.ov_feat_post_layer = nn.Sequential(
            nn.Linear(self.ov_feat_dim, self.ov_feat_dim),
            nn.ReLU(),
            nn.Linear(self.ov_feat_dim, self.ov_feat_dim))

        # for debug
        self.count = 0

    def init_weights(self):
        pass

    def forward_backbone(self, imgs, extra_feat=None):
        predictions = dict()

        if self.use_depthanything:
            output = self.backbone.custom_forward(imgs.flatten(0, 1))
            return output

    def prepare_gaussian_args(self, gaussians, metas):

        means = gaussians.means # b, g, 3
        # myfix
        b_, g_, _ = means.shape
        # means = means.reshape(-1, 3)
        # means_cam = torch.cat((means, torch.ones((means.shape[0], 1), device=means.device)), dim=1).to(torch.float32)
        means_cam = F.pad(means, (0, 1), value=1)
        # cam2world = metas[0]['cam2world'].to(torch.float32)
        cam2world = torch.stack([meta['cam2world'] for meta in metas]).float().cuda()
        # means_world_ = (cam2world @ means_cam.unsqueeze(-1)).squeeze(-1)
        means_world_ = einsum(cam2world, means_cam, 'b n k, b j k -> b j n')
        means_world = means_world_[..., :3]

        # means_world = means_world.reshape(b_, g_, 3)
        means = means_world
        # endfix
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1

        if opacities is None:
            opacities = torch.ones_like(origi_opa)

        # if origi_opa.numel() == 0:
        #     origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)

        bs, g, _ = means.shape

        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]

        R = get_rotation_matrix(rotations) # b, g, 3, 3

        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)

        c2w_rot = torch.stack([meta['cam2world'][:3, :3] for meta in metas]).cuda()
        # c2w_rot_T = torch.stack([meta['cam2world'][:3, :3].T for meta in metas])
        c2w_rot_T = rearrange(c2w_rot, 'b h w -> b w h')
        # use expand to avoid materializing B*G copies in memory
        c2w_rot = c2w_rot.unsqueeze(1).expand(-1, g, -1, -1).to(torch.float32)
        c2w_rot_T = c2w_rot_T.unsqueeze(1).expand(-1, g, -1, -1).to(torch.float32)
        Cov = torch.matmul(c2w_rot, torch.matmul(Cov, c2w_rot_T))
        # endfix

        # CovInv = Cov.float().cpu().inverse().cuda() # b, g, 3, 3
        CovInv = Cov.double().inverse().to(Cov.dtype)
        return means, origi_opa, opacities, scales, CovInv

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        extra_feat=None,
        return_gaussian=False,
        global_training_ratio=1.,
        test_prompt=None,
        **kwargs,
    ):
        # label = label.flatten(0, 1)

        assert imgs.shape[2] == 1, f'#view == 1, but got {imgs.shape[2]}'
        imgs = imgs.squeeze(2)
        imgs = rearrange(imgs, 'b f h w c -> b f c h w')

        # keep fm_feats available for downstream losses (feature_alignment_loss)
        with torch.no_grad():
            fm_feats = self.get_fm_feat(self.ov_model, imgs, metas)

        nyu_pc_min = torch.stack([meta['vox_origin'] for meta in metas])
        scene_size = torch.stack([meta['scene_size'] for meta in metas])
        nyu_pc_max = nyu_pc_min + scene_size

        sampled_xyz = torch.stack([meta['occ_xyz'] if isinstance(meta['occ_xyz'], torch.Tensor) else torch.from_numpy(meta['occ_xyz']).cuda() for meta in metas])

        if self.use_depthanything:
            imgs = (imgs - self._resnet_mean) / self._resnet_std

        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        if self.frozen_backbone:
            with torch.no_grad():
                self.backbone.eval()
                predictions = self.forward_backbone(imgs, extra_feat=extra_feat)
        else:
            predictions = self.forward_backbone(imgs, extra_feat=extra_feat)

        mlvl_img_feats = []
        if self.use_depthanything:
            mlvl_img_feats = [t[0] for t in predictions[0]]
            ori_anchor_feat = self.gs_head.custom_forward(*predictions)
            ori_anchor_feat = ori_anchor_feat.unflatten(0, imgs.shape[:2])
        else:
            assert False

        anchor_feat = self.anchor_feat_layer(ori_anchor_feat.flatten(0, 1))
        anchor_depth = self.anchor_depth_pred(anchor_feat)

        # TODO
        # anchor_depth = inverse_log_transform(anchor_depth)
        max_depth = 8
        anchor_depth = anchor_depth.sigmoid() * max_depth  # (B, 1, H, W)

        device = anchor_depth.device
        dtype = anchor_depth.dtype
        B, _, H, W = anchor_depth.shape
        iH, iW = imgs.shape[-2:]

        # Step 2: Generate pixel coordinate grid (u, v)
        u, v = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )  # (H, W)
        uv = torch.stack([u, v, torch.ones_like(u)], dim=0).float()  # (3, H, W)
        uv = uv.unsqueeze(0).expand(B, -1, -1, -1).to(dtype)  # (B, 3, H, W)

        # Step 3: Parse camera intrinsics
        intr = torch.stack([meta['cam_k'] for meta in metas]).to(dtype).to(device)  # (B, 3, 3)
        fx = intr[:, 0, 0].view(B, 1, 1)
        fy = intr[:, 1, 1].view(B, 1, 1)
        cx = intr[:, 0, 2].view(B, 1, 1)
        cy = intr[:, 1, 2].view(B, 1, 1)

        # Step 4: Compute normalized viewing directions in camera coordinates (B, 3, H, W)
        x = (uv[:, 0] / W * iW - cx) / fx
        y = (uv[:, 1] / H * iH - cy) / fy
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=1)
        # dirs = torch.nn.functional.normalize(dirs, dim=1, eps=1e-6)  # unit direction vectors

        # Step 5: Back-project pixels to 3D points using depth (B, 3, H, W)
        xyz = dirs * anchor_depth  # element-wise scale by depth

        if self.num_bins > 0:

            # Step 6: Create per-pixel depth offsets (scaled by predicted scale)
            depth_scale = self.depth_scale_layer(anchor_feat).sigmoid()  # (B, num_bins, H, W)
            depth_offset = torch.linspace(
                0, 5, self.num_bins, device=device, dtype=dtype
            )[None, :, None, None] * depth_scale  # (B, num_bins, H, W)

            Z_bins = anchor_depth + depth_offset
            x_ = repeat(x, 'b h w -> b k h w', k=self.num_bins)
            y_ = repeat(y, 'b h w -> b k h w', k=self.num_bins)

            X_bins = x_ * Z_bins
            Y_bins = y_ * Z_bins
            xyz_offset = torch.stack([X_bins, Y_bins, Z_bins], dim=2)  # (B, nb, 3, H, W)

            anchor_feat_offset = anchor_feat.unsqueeze(1) + self.bins_emb.weight[None, :, :, None, None]

            # softrelu
            # semantics = F.softplus(semantics)
        else:
            xyz_offset = xyz[:, None]
            anchor_feat_offset = anchor_feat[:, None]

        xyz = rearrange(xyz_offset, 'b n c h w -> b (n h w) c')
        feat = rearrange(anchor_feat_offset, 'b n c h w -> b (n h w) c')
        gs_feat = self.gs_pred_layer(feat)
        query = self.feat2query(feat)

        ref_pt = torch.stack([u / W, v / H], dim=-1).float()
        ref_pt = repeat(ref_pt, 'h w c -> b (n h w) l c',
                        b=B, n=self.num_bins if self.num_bins > 0 else 1,
                        l=len(mlvl_img_feats))
        query_pos = self.xyz2qpos(xyz)

        device = query.device
        # [0] is the current frame

        fh, fw = imgs.shape[-2] // self.patch_size, imgs.shape[-1] // self.patch_size
        num_lvl = len(mlvl_img_feats)
        spatial_shapes = torch.as_tensor([[fh, fw]] * num_lvl, dtype=torch.long, device=device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ), 0)

        kv = torch.cat(mlvl_img_feats, dim=1)
        kv = self.kv_proj(kv)
        del mlvl_img_feats

        ov_feat = self.deform_attn(
            query=query,
            key=kv, value=kv,
            query_pos=query_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=ref_pt,
        )

        ov_feat = self.ov_feat_post_layer(ov_feat)
        del kv, query_pos, ref_pt, spatial_shapes, level_start_index

        gs_scales = safe_sigmoid(gs_feat[..., :3])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = gs_feat[..., 3:7]

        if self.min_max_temperature:
            if self.training:
                T_min = self.min_max_temperature[0]
                T_max = self.min_max_temperature[1]
                p = global_training_ratio  # [0,1]
                temperature = max(T_min, T_max * (T_min / T_max) ** p)
            else:
                temperature = self.min_max_temperature[0]
        else:
            temperature = 1.

        logit_opacity = gs_feat[..., 7:8]

        assert self.with_opacity
        shs = torch.zeros(*gs_feat.shape[:-1], 0, device=device, dtype=dtype)

        opas = torch.sigmoid(logit_opacity / temperature)

        if self.semantic_dim > 0:
            semantics = gs_feat[..., (8 if self.with_opacity else 7) : (8 if self.with_opacity else 7) + self.semantic_dim]
        else:
            semantics = None

        # for render
        cam2img = torch.from_numpy(np.stack([meta['cam2img'] for meta in metas])).to(imgs)
        image_size = imgs.shape[-2:]
        ds_cam2img = cam2img
        ds_image_size = image_size

        cur_cam2ego = torch.stack([meta['cam2world'] for meta in metas]).to(xyz)
        if self.training:
            nei_cam2ego = torch.from_numpy(np.stack([np.stack(meta['neighbor_cam2world']) for meta in metas])).to(xyz)
            cam2ego = torch.cat([cur_cam2ego[:, None], nei_cam2ego], dim=1)
        else:
            cam2ego = cur_cam2ego[:, None]
        ds_cam2img = repeat(ds_cam2img, 'b ... -> b n ...', n=cam2ego.shape[1])

        conf = None

        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            # harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics,
            feat=ov_feat,
            conf=conf
        )
        if return_gaussian:
            return dict(gaussian=gaussian), None, None

        result_dict = {'fm_feats': fm_feats}

        sparse_gaussians = []
        sparse_occ_preds = []
        sparse_labels = []
        sparse_masks = []

        # cam to world
        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussian, metas)

        resize_imgs = F.interpolate(imgs.flatten(0, 1), size=(H, W), mode='bicubic', align_corners=False)
        ov_rgbs = rearrange(repeat(resize_imgs, 'b c h w -> b c n h w', n=self.num_bins if self.num_bins > 0 else 1), 'b c n h w -> b (n h w) c')
        ov_feat_to_render = torch.cat([ov_feat, ov_rgbs], dim=-1)

        if self.with_unc and self.training:
            ov_feat_to_render = torch.cat([ov_feat_to_render, conf], dim=-1)


        rendered, alpha = rasterize_gaussians(
            means,
            # F.pad(inst_feats, (0, 1), value=1), # pad one for density rendering
            # torch.cat([inst_feats, 1 - occupied_logits.sigmoid()], dim=-1),
            ov_feat_to_render,
            origi_opa.squeeze(-1),
            gs_scales, # * self.aggregator.scale_multiplier,
            rot,
            ds_cam2img, # [:, None], # multi view
            cam2ego, # cam2ego[:, None], # multi view
            img_aug_mats=None, # multi view
            image_size=ds_image_size, # all batch same shape
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32,
            # tile_size=16, # default 16
        ) # .flatten(0, 1)

        rendered, rendered_depth = rendered[:, :, :-1], rendered[:, :, -1]

        # if self.with_unc and self.training:
        #     rendered, rendered_conf = torch.split(rendered, (rendered.shape[2] - 1, 1), dim=2)
        # else:
        rendered_conf = None

        rendered, rendered_rgb = torch.split(rendered, (rendered.shape[2] - 3, 3), dim=2)

        result_dict.update({
            'rendered_depth': rendered_depth,
            'rendered_first_depth': rendered_depth[..., 0:1],
            'rendered': rendered,
            'rendered_alpha': alpha,
            'rendered_first_alpha': alpha[..., 0:1],
            'rendered_rgb': rendered_rgb,
            # 'rendered_density': rendered_density
            'rendered_conf': rendered_conf,
        })
        if self.training:
            neighbor_rgbs = torch.stack([
                F.interpolate(
                    rearrange(meta['neighbor_rgbs'], 'n h w c -> n c h w'),
                    size=rendered.shape[-2:], mode='bicubic', align_corners=False
                )
                for meta in metas
            ]).cuda()
            if self.use_depthanything:
                neighbor_rgbs = (neighbor_rgbs - self._resnet_mean) / self._resnet_std

            rgb_gt = torch.cat([imgs, neighbor_rgbs], dim=1)
            result_dict['rgb_gt'] = rgb_gt

        semantics = []
        occ_ov_feats = []
        densities = []
        sem_labels = []
        fov_mask = []
        confs = []

        # if self.with_unc:
        #     opacities = torch.cat([opacities, conf], dim=-1)

        for i in range(len(sampled_xyz)):
            epsilon = 1e-3
            mask = (
                means[i, ..., 0] > (nyu_pc_min[i, 0]+epsilon)
            ) & (
                means[i, ..., 0] < (nyu_pc_max[i, 0]-epsilon)
            ) & (
                means[i, ..., 1] > (nyu_pc_min[i, 1]+epsilon)
            ) & (
                means[i, ..., 1] < (nyu_pc_max[i, 1]-epsilon)
            ) & (
                means[i, ..., 2] > (nyu_pc_min[i, 2]+epsilon)
            ) & (
                means[i, ..., 2] < (nyu_pc_max[i, 2]-epsilon)
            )

            # print(mask.sum())

            if self.training and mask.sum() < 100:
                print(f'skip due to only {mask.sum()} points inside range')
                continue

            if not self.training:
                opacities = torch.cat([ov_feat, opacities], dim=-1)

            if not self.training:
                mask = mask & (origi_opa[i].squeeze(-1) > self.opacities_threshold)

            if not self.training and mask.sum() == 0:
                semantic = torch.zeros(
                    *sampled_xyz[i:(i+1)].shape[:4],
                    opacities.shape[-1],
                    device=opacities.device,
                    dtype=opacities.dtype).flatten(0, -2)
                bin_logit = torch.zeros_like(semantic[..., 0])
                density = torch.zeros_like(semantic[..., 0])
            else:
                semantic, bin_logit, density = self.aggregator(
                    sampled_xyz[i:(i+1)].flatten(1, 3),
                    means[i][mask][None],
                    origi_opa[i][mask][None],
                    opacities[i][mask][None],
                    scales[i][mask][None],
                    CovInv[i][mask][None],
                    metas[i],
                    nyu_pc_min[i]) # n, c

            if not self.training:
                occ_ov_feat, semantic = torch.split(semantic, [
                    ov_feat.shape[-1],
                    opacities.shape[-1]-ov_feat.shape[-1]
                ], dim=-1)
                occ_ov_feats.append(occ_ov_feat)

            # if self.with_unc:
            #     sem = semantic[:, :-2] * bin_logit.unsqueeze(-1)
            #     occ_conf = semantic[:, -2]
            #     confs.append(occ_conf)
            # else:
            sem = semantic[:, :-1] * bin_logit.unsqueeze(-1)

            geo = 1 - bin_logit.unsqueeze(-1)
            geosem = torch.cat([sem, geo * self.bin_logit_scale], dim=-1)

            # semantics.append(1 - bin_logit.unsqueeze(-1))
            semantics.append(geosem)
            densities.append(density)

            sem_labels.append(label[i])
            fov_mask.append(metas[i]['fov_mask'])

        result_dict.update(dict(
            depth_pred=anchor_depth,
            depth_gt=torch.stack([meta['depth_gt'] for meta in metas]),
        ))

        if len(semantics):
            semantics = torch.stack(semantics, dim=0).transpose(1, 2) # [1, 13, 129600]
            spatial_shape = label.shape[2:] # [60, 60, 36]

            result_dict.update({
                'ce_input': semantics.unflatten(-1, spatial_shape),
                'ce_label': torch.stack(sem_labels).flatten(0, 1),
                'fov_mask': torch.stack(fov_mask).bool(),
            })

            if self.with_unc:
                result_dict.update(dict(confs=confs))

            if not self.training:
                occ_ov_feats = torch.stack(occ_ov_feats)
                occ_ov_feats = rearrange(
                    occ_ov_feats, 'b (h w d) c -> b c h w d',
                    h=spatial_shape[0], w=spatial_shape[1], d=spatial_shape[2])

                lang_feat = self.test_lang_feat(test_prompt)
                result_dict['lang_feat'] = lang_feat

                occ_ov_cos_sim = einsum(
                    occ_ov_feats / occ_ov_feats.norm(dim=1, keepdim=True),
                    lang_feat, 'b c h w d, k c -> b k h w d')
                result_dict.update(dict(
                    occ_ov_cos_sim=occ_ov_cos_sim,
                    classnames=self.ov_model.query_words))

        result_dict.update({
            'gaussian': gaussian,
        })

        return result_dict, None, None
