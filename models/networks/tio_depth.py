import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19

from models.backbones.resnet import ResNet_Backbone
from models.backbones.swin import get_orgwintrans_backbone
from models.base_model import Base_of_Model
from models.decoders.dual_path_decoder import DPDecoder
from utils import platform_manager


@platform_manager.MODELS.add_module
class TiO_Depth(Base_of_Model):
    def _initialize_model(
            self,
            encoder_name='Res50',
            decoder_ch_num = [16, 32, 64, 128, 256],
            min_disp=1.6,
            max_disp=240, 
            d2d=320.72,
            image_size=[192, 640],
            set_train_side='s',
            downscale_occ=False,
            stereo_train_sides=['s'],
        
            decoder_name='',
            out_ch=1,
            out_mode='Mono',
            set_fuse_mode='Add',
            
            discrete_warp=False,
            discrete_warp_scale=1,
            params_trained='',
            set_SCALE=None,

    ):
        self.init_opts = locals()

        self.decoder_ch_num = decoder_ch_num
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.d2d = d2d
        self.image_size = image_size
        self.set_train_side = set_train_side
        self.downscale_occ = downscale_occ
        self.stereo_train_sides = stereo_train_sides
        self.decoder_name = decoder_name
        self.out_ch = out_ch
        self.out_mode = out_mode
        self.set_dec_mode = set_fuse_mode
        self.discrete_warp = discrete_warp
        self.discrete_warp_scale = discrete_warp_scale
        self.params_trained = params_trained
        self.set_SCALE = set_SCALE
        self.stereo_SCALE = 5.4 if (self.out_ch == 1) else 1

        self.net_module = {}
        
        if out_ch > 1:
            out_range = []
            rel_disp = max_disp / min_disp
            for disp_idx in range(self.out_ch):
                index_num = rel_disp**(disp_idx / (self.out_ch - 1) - 1)
                disp = max_disp * index_num
                out_range.append(disp)
            volume_dk = torch.tensor(out_range).unsqueeze(1).unsqueeze(1)
            volume_dk = volume_dk.unsqueeze(0).unsqueeze(0)
            self.volume_dk = volume_dk.to(self.device)
        
        if 'orgSwin' in encoder_name:
            self.net_module['encoder'], enc_ch_num = get_orgwintrans_backbone(
                encoder_name, True)
            enc_ch_num = copy.deepcopy(enc_ch_num)
        elif 'Res' in encoder_name:
            layer_num = int(encoder_name[3:])
            self.net_module['encoder'] = ResNet_Backbone(layer_num=layer_num)
            enc_ch_num = [64, 64, 128, 256, 512]
            if layer_num > 34:
                enc_ch_num = [ch_num * 4 for ch_num in enc_ch_num]
                enc_ch_num[0] = 64

        sfg_scales = [int(n) if len(n) == 1 else int(n[0]) for n in decoder_name.split('-')[1:]]
        db_scales = [int(n) if len(n) == 1 else int(n[0]) for n in decoder_name.split('_')[1:]]
        self.db_scales = db_scales
        sfg_mode = 'MFM'
        if 'Attn' in decoder_name:
            sfg_mode = 'Attn'
        if 'MFM' in decoder_name:
            sfg_mode = 'MFM'
        if 'CVGCat' in decoder_name:
            sfg_mode = 'Cat'
        
        db_mode = 'SDFA'
        self.use_db_mode = 'Out'
        if 'DeformConv' in decoder_name:
            db_mode = 'DeformConv'
        elif 'FinalBranch' in decoder_name:
            db_mode = 'SDFA'

        if 'SDFARaw' in decoder_name:
            dec_mode = 'SDFA'
        else:
            dec_mode = ''

        stereo_out_ch = out_ch
        
        if 'WoOutL' in decoder_name:
            disable_disbranch = True
        else:
            disable_disbranch = False

        self.net_module['decoder'] = DPDecoder(enc_ch_num,
                                                    [disp /image_size[1] for disp in out_range],
                                                    num_ch_dec=decoder_ch_num,
                                                    sfg_scales=sfg_scales,
                                                    sfg_mode=sfg_mode,
                                                    db_scales=db_scales,
                                                    db_mode=db_mode,
                                                    dec_mode = dec_mode,
                                                    stereo_out_ch=stereo_out_ch,
                                                    mono_out_ch=out_ch,
                                                    mid_out_ch=out_ch,
                                                    disable_disbranch = disable_disbranch,
                                                    image_size=image_size)
        self.use_midout = len(db_scales) >= 1

        self.enc_ch_num = enc_ch_num

        self._networks = nn.ModuleList(list(self.net_module.values()))

        self.projector = {}
        self.projector['c'] = DepthProjector(image_size).to(self.device)
        
        if self.discrete_warp:
            self.projector['d'] = DepthTransformer([s // self.discrete_warp_scale for s in image_size],
                                                   [d2d / disp for disp in out_range],
                                                   self.device)
            self.feat_net = Feat_Net(net_mode='vgg19', device=self.device)
        
        self.used_path_dict = ['Mono', 'Stereo', 'Refine']
                              
    def forward(self, x, outputs, **kargs):
        x = (x - 0.45) / 0.225
        if self.is_train:
            used_path = self.used_path_dict[self.now_group_idx]
        else:
            if hasattr(self, 'used_out_mode'):
                used_path = self.used_out_mode
            else:
                used_path = self.out_mode[0]
        if used_path == 'Mono':
            t_side = self.set_train_side if self.is_train else 's'
            feats = self.net_module['encoder'](x)
            outputs['enc_feats_{}'.format(t_side)] = feats
            outputs['enc_feats_{}_wog'.format(t_side)] =\
                [feat.detach() for feat in feats]
        
            outputs = self._forward_decoder_mono(outputs,
                                                 feats,
                                                 x.shape,
                                                 t_side)
            pred = outputs['mono_depth_0_{}'.format(t_side)]
            outputs['rawdepth'] = pred
            pred_disp = outputs['mono_disp_0_{}'.format(t_side)]
            outputs['rawdisp'] = pred_disp
            
        elif used_path == 'Stereo':
            # TODO check no grad
            with torch.no_grad():
                if 'enc_feats_s_wog' not in outputs:
                    feats = self.net_module['encoder'](x[:, :3, ...])
                    outputs['enc_feats_s_wog'] = feats
                    outputs = self._forward_decoder_mono(outputs,
                                                         feats,
                                                         x.shape,
                                                         's')
                if 'enc_feats_o_wog' not in outputs:
                    feats = self.net_module['encoder'](x[:, 3:, ...])
                    outputs['enc_feats_o_wog'] = feats
                    outputs = self._forward_decoder_mono(outputs,
                                                         feats,
                                                         x.shape,
                                                         'o')
        
            if self.inputs and 'direct' in self.inputs:
                directs = self.inputs['direct']
            else:
                directs = torch.tensor([1]).to(x)
            outputs = self._forward_decoder_stereo(outputs,
                                                   x.shape,
                                                   directs)
            pred = outputs['stereo_depth_0_s']
            outputs['sdepth'] = pred
            pred_disp = outputs['stereo_disp_0_s']
            outputs['sdisp'] = pred_disp

        elif used_path == 'Refine':
            t_side = 's'
            if 'enc_feats_{}_wog'.format(t_side) not in outputs:
                feats = self.net_module['encoder'](x)
                outputs['enc_feats_{}_wog'.format(t_side)] = feats
            else:
                feats = outputs['enc_feats_{}_wog'.format(t_side)]
            outputs = self._forward_decoder_mono(outputs,
                                                 feats,
                                                 x.shape,
                                                 t_side,
                                                 with_mo=self.use_midout,
                                                 name='ref')
            pred = outputs['refmono_depth_0_{}'.format(t_side)]
            outputs['refdepth'] = pred
            pred_disp = outputs['refmono_disp_0_{}'.format(t_side)]
            outputs['refdisp'] = pred_disp
        
        outputs[('depth', 's')] = pred
        outputs['disp', 's'] = pred_disp
        return pred, outputs

    def _preprocess_inputs(self):
        if self.is_train:
            used_path = self.used_path_dict[self.now_group_idx]
            aug = '_aug'
        else:
            if hasattr(self, 'used_out_mode'):
                used_path = self.used_out_mode
            else:
                used_path = self.out_mode
            aug = ''
        if used_path == 'Mono':
            t_side = self.set_train_side if self.is_train else 's'
            x = self.inputs['color_{}{}'.format(t_side, aug)]

        elif used_path == 'Stereo':
            x_s = self.inputs['color_s{}'.format(aug)]
            x_o = self.inputs['color_o{}'.format(aug)]
            x = torch.cat([x_s, x_o], dim=1)
        
        else:
            x = self.inputs['color_s{}'.format(aug)]
        
        return x

    def _postprocess_outputs(self, outputs):
        used_path = self.used_path_dict[self.now_group_idx]
        if used_path == 'Mono':
            t_side = self.set_train_side
            loss_sides = [t_side]
            outputs = self._forward_proj(outputs, 'mono', t_side)
        elif used_path == 'Stereo':
            for t_side in self.stereo_train_sides:
                outputs = self._forward_proj(outputs, 'stereo', t_side)
            loss_sides = self.stereo_train_sides
        elif used_path == 'Refine':
            t_side = 's'
            loss_sides = [t_side]
            mono_depth = outputs['mono_depth_0_{}'.format(t_side)].detach()
            stereo_depth = outputs['stereo_depth_0_{}'.format(t_side)].detach()  * self.stereo_SCALE
            if self.set_dec_mode == 'Add':
                fuse_depth = (mono_depth + stereo_depth) / 2
            elif self.set_dec_mode == 'OccLap':
                mask_builder = SelfOccluMask(1, 60, device=self.device)
                if self.downscale_occ:
                    stereo_disp = outputs['stereo_disp_0_{}'.format(t_side)].detach() / 2
                    stereo_disp = F.interpolate(stereo_disp, [s // 2 for s in stereo_disp.shape[2:]], mode='bilinear', align_corners=False)
                    occ_mask = mask_builder(stereo_disp, self.inputs['direct'] * (1 if t_side == 's' else -1))
                    occ_mask = F.interpolate(occ_mask, [s * 2 for s in occ_mask.shape[2:]], mode='bilinear', align_corners=False)
                plane_builder = LapMask()
                _, plane_mask = plane_builder(stereo_depth)
                plane_mask[occ_mask < 0.8] = 1
                fuse_depth = stereo_depth * plane_mask + mono_depth * (1 - plane_mask)
                outputs['plane_mask_{}'.format(t_side)] = plane_mask
            
            outputs['fuse_depth_{}'.format(t_side)] = fuse_depth
            outputs['fuse_disp_{}'.format(t_side)] = self.d2d / fuse_depth

            mono_volume = outputs['mono_pvolume_0_{}'.format(t_side)].detach()
            stereo_volume = outputs['stereo_pvolume_0_{}'.format(t_side)].detach()
            if self.set_dec_mode == 'Add':
                fuse_volume = (mono_volume + stereo_volume) / 2
            elif self.set_dec_mode == 'OccLap':
                fuse_volume = mono_volume * (1 - plane_mask) + stereo_volume *  plane_mask
            
            outputs['fuse_pvolume_{}'.format(t_side)] = fuse_volume

            K = self.inputs['K'].clone()
            inv_K = self.inputs['inv_K']
            frames = [t_side.replace('s', 'o') if 's' in t_side\
                else t_side.replace('o', 's')]
            for id_frame in frames:
                source_name = 'color_{}'.format(id_frame)
                source_img = self.inputs[source_name]
                T = self.inputs['T'].clone()
                if 'o' in id_frame:
                    T[:, 0, 3] = T[:, 0, 3]
                else: # id_frame == 's'
                    T[:, 0, 3] = -T[:, 0, 3]
                
                projecter_res = self.projector['c'](fuse_depth, inv_K, T, K,
                    source_img, False)
                projected_img = projecter_res[0]
                outputs['fuse_proj_img_{}_{}_{}'.format(
                    id_frame, 0, t_side)] = projected_img
                
                refmono_depth = outputs['refmono_depth_0_{}'.format(t_side)]
                projecter_res_m = self.projector['c'](refmono_depth, inv_K, T, K,
                    source_img, False)
                projected_img_m = projecter_res_m[0]
                outputs['refmono_proj_img_{}_{}_{}'.format(
                    id_frame, 0, t_side)] = projected_img_m
            
                mask_builder = SelfOccluMask(1, 60, device=self.device)
                fuse_disp = outputs['fuse_disp_{}'.format(t_side)].detach()
                fuse_occ_mask = mask_builder(fuse_disp, self.inputs['direct'] * (-1 if t_side == 's' else 1))
                fuse_occ_mask = (fuse_occ_mask > 0.5).to(torch.float)

                # compute fuse image
                target_name = 'color_{}'.format(t_side)
                raw_img = self.inputs[target_name]
                refmono_img = projected_img_m
                fused_fuse_mono = fuse_occ_mask * raw_img + (1 - fuse_occ_mask) * refmono_img
                outputs['fused_fuse_occ_color_{}'.format(t_side)] = fused_fuse_mono
                outputs['fuse_occ_mask_{}'.format(t_side)] = fuse_occ_mask
                
        return loss_sides, outputs
        

    def _forward_decoder_mono(self, outputs, feats, img_shape, t_side, with_mo=False, name=''):
        depth_scale = img_shape
        raw_outputs =  self.net_module['decoder'](feats, depth_scale, with_mo=with_mo)
        
        mono_outputs = raw_outputs[0]
        volume_dk = self.volume_dk
        d2d = self.d2d
        
        if with_mo:
            if self.use_db_mode == 'Out':
                mono_outputs = raw_outputs[0]['0-mid']
            else:
                for i in self.db_scales:
                    if i == max(self.db_scales):
                        mid_outputs = raw_outputs[0]['{}-mid'.format(i)]
                    else:
                        mid_outputs += raw_outputs[0]['{}-mid'.format(i)]
                
                if self.use_db_mode == 'Add':
                    mono_outputs = mid_outputs
                else:
                    mono_outputs = mono_outputs[0]

        # compute output depth
        d_volume = mono_outputs.unsqueeze(1)
        p_volume = F.softmax(d_volume, dim=2)
        if with_mo and self.use_db_mode == 'ResV':
            res_d_volume = mid_outputs.unsqueeze(1)
            res_p_volume = F.softmax(res_d_volume, dim=2)
            p_volume = (p_volume.detach() + res_p_volume) / 2
        disp = (p_volume * volume_dk).sum(dim=2)
        depth = d2d / disp
        if with_mo and self.use_db_mode == 'ResD':
            residual = mid_outputs[:, 0:2, ...]
            norm_residual = F.softmax(residual, dim=1)
            depth_residual = norm_residual[:, 0:1, ...] * -5 + norm_residual[:, 1:2, ...] * 5
            depth = (depth.detach()+depth_residual).clamp(self.min_depth, self.max_depth)
            outputs['{}_residual_{}'.format(name+'mono', t_side)] = depth_residual

        outputs['{}_dvolume_{}_{}'.format(name+'mono', 0, t_side)] = d_volume.squeeze(1)
        outputs['{}_pvolume_{}_{}'.format(name+'mono', 0, t_side)] = p_volume.squeeze(1).detach()
        outputs['{}_pvolume_wg_{}_{}'.format(name+'mono', 0, t_side)] = p_volume.squeeze(1)
        outputs['{}_disp_{}_{}'.format(name+'mono', 0, t_side)] = disp
        outputs['{}_depth_{}_{}'.format(name+'mono', 0, t_side)] = depth

        for k in raw_outputs[1].keys():
            outputs['{}_{}_{}'.format(name+'mono', k, t_side)] = raw_outputs[1][k]

        return outputs

    def _forward_decoder_stereo(self, outputs, img_shape, directs):
        depth_scale = img_shape
        
        features = []
        for f_idx in range(len(outputs['enc_feats_s_wog'])):
            features.append([outputs['enc_feats_s_wog'][f_idx],
                             outputs['enc_feats_o_wog'][f_idx]])
        
        directs = -directs
        directs = directs.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        raw_outputs = self.net_module['decoder'](
            features, depth_scale, directs, out_two_side=True)

        costs = raw_outputs[1]
        for k, v in costs.items():
            d_volume = v.unsqueeze(1)
            p_volume = F.softmax(d_volume, dim=2)
            outputs['cost_{}'.format(k)] = p_volume.squeeze(1)
            disp = (p_volume * self.volume_dk).sum(dim=2)
            outputs['cost_disp_{}'.format(k)] = disp
        
        for out_side in ['s', 'o']:
            stereo_outputs = raw_outputs[0]['0-{}'.format(out_side)]
            if stereo_outputs.shape[1] == 1:
                norm_disp = torch.sigmoid(stereo_outputs)
                _, depth = self._disp2depth(norm_disp)
                disp = self.d2d / depth
            else:
                volume_dk = self.volume_dk
                d2d = self.d2d
                # compute output depth
                d_volume = stereo_outputs.unsqueeze(1)
                p_volume = F.softmax(d_volume, dim=2)
                disp = (p_volume * volume_dk).sum(dim=2)
                depth = d2d / disp
                outputs['{}_dvolume_{}_{}'.format('stereo', 0, out_side)] =\
                    d_volume.squeeze(1)
     
                outputs['{}_pvolume_{}_{}'.format('stereo', 0, out_side)] = \
                    p_volume.squeeze(1).detach()
            outputs['{}_disp_{}_{}'.format('stereo', 0, out_side)] = disp
            outputs['{}_depth_{}_{}'.format('stereo', 0, out_side)] = depth

        return outputs

    def _forward_proj(self, outputs, out_name, t_side):
        K = self.inputs['K'].clone()
        inv_K = self.inputs['inv_K']
        
        if out_name == 'mono':
            d_volume = outputs['{}_dvolume_{}_{}'.format('mono', 0, t_side)]
            d_volume = d_volume.unsqueeze(1)
            # reproject image
            frames = [t_side.replace('s', 'o') if 's' in t_side\
                else t_side.replace('o', 's')]
            for id_frame in frames:
                T = self.inputs['T'].clone()
                if 'o' in id_frame:
                    T[:, 0, 3] = -T[:, 0, 3]
                else: # id_frame == 's'
                    T[:, 0, 3] = T[:, 0, 3]
                            
                source_name = 'color_{}_aug'.format(t_side)
                target_name = 'color_{}_aug'.format(id_frame)
                source_img = self.inputs[source_name]
                raw_img = self.inputs[target_name]

                if self.discrete_warp_scale > 1:
                    warp_size = [s // self.discrete_warp_scale for s in d_volume.shape[3:]]
                    d_volume = F.interpolate(d_volume.squeeze(1), warp_size, mode='bilinear', align_corners=False)
                    d_volume = d_volume.unsqueeze(1)
                    source_img = F.interpolate(source_img, warp_size, mode='bilinear', align_corners=False)
                    raw_img = F.interpolate(raw_img, warp_size, mode='bilinear', align_corners=False)
                    self.inputs['color_{}_1_aug'.format(t_side)] = source_img
                    self.inputs['color_{}_1_aug'.format(id_frame)] = raw_img
                    K[:, :2,:] /= self.discrete_warp_scale
                    inv_K = torch.from_numpy(np.linalg.pinv(K.cpu().numpy())).to(K)

                w_d_volume = self.projector['d']\
                    .get_warped_volume(d_volume, inv_K, T, K)
                w_p_volume = F.softmax(w_d_volume, dim=2)
                w_img = self.projector['d']\
                    .get_warped_frame(source_img, inv_K, T, K)
                synth_img = (w_p_volume * w_img).sum(dim=2)
                
                outputs['{}_disproj_img_{}_{}_{}'.format(
                    out_name, id_frame, 0, t_side)] = synth_img

                with torch.no_grad():
                    raw_feats = self.feat_net.get_feats(raw_img)
                synth_feats = self.feat_net.get_feats(synth_img)
                for feat_idx in range(3):
                    rawf_name = 'mono_raw_feats_{}_{}_{}'.format(
                        id_frame, feat_idx, t_side)
                    outputs[rawf_name] = raw_feats[feat_idx]
                    synthf_name = 'mono_synth_feats_{}_{}_{}'.format(
                        id_frame, feat_idx, t_side)
                    outputs[synthf_name] = synth_feats[feat_idx] 
        
        if out_name == 'stereo':
            # get cost guide
            tmp_cost_scales = [int(n) if len(n) == 1 else int(n[0]) for n in self.decoder_name.split('-')[1:]]
            tmp_p_volume = outputs['mono_pvolume_0_{}'.format(t_side)].squeeze(1)
            for i in range(1, 5):
                tmp_p_volume = F.interpolate(tmp_p_volume, [s//2 for s in tmp_p_volume.shape[2:]],
                                                mode='bilinear', align_corners=True)
                outputs['{}_pvolume_cost_{}_{}'.format('mono', i, t_side)] = tmp_p_volume
    
            # reproject image
            depth = outputs['{}_depth_{}_{}'.format('stereo', 0, t_side)] * self.stereo_SCALE
            frames = [t_side.replace('s', 'o') if 's' in t_side\
                else t_side.replace('o', 's')]
            for id_frame in frames:
                source_name = 'color_{}'.format(id_frame)
                target_name = 'color_{}'.format(t_side)
                source_img = self.inputs[source_name]
                T = self.inputs['T'].clone()
                if 'o' in id_frame:
                    T[:, 0, 3] = T[:, 0, 3]
                else: # id_frame == 's'
                    T[:, 0, 3] = -T[:, 0, 3]
                projecter_res = self.projector['c'](depth, inv_K, T, K,
                    source_img, False)
                projected_img = projecter_res[0]
                outputs['{}_proj_img_{}_{}_{}'.format(
                    out_name, id_frame, 0, t_side)] = projected_img

                if 'mono_depth_0_{}'.format(t_side) in outputs:
                    if 'mono_proj_img_{}_0_{}'.format(id_frame, t_side) in outputs:
                        m_projected_img =  outputs['mono_proj_img_{}_0_{}'
                            .format(id_frame, t_side)].detach()
                    else:
                        mono_depth = outputs['mono_depth_0_{}'.format(t_side)]
                        m_projecter_res = self.projector['c'](
                            mono_depth, inv_K, T, K, source_img, True)
                        m_projected_img = m_projecter_res[0]
                        edge_mask = m_projecter_res[1]
                        outputs['edge_mask_{}'.format(t_side)] = edge_mask
                        outputs['inv_edge_mask_{}'.format(t_side)] = 1 - edge_mask
                        outputs['mono_proj_img_{}_0_{}'.format(id_frame, t_side)] = m_projected_img

                    # compute occlusion mask
                    mask_builder = SelfOccluMask(1, 60, device=self.device)
                    if self.downscale_occ:
                        mono_disp = outputs['mono_disp_0_{}'.format(t_side)].detach() / 2
                        mono_disp = F.interpolate(mono_disp, [s // 2 for s in mono_disp.shape[2:]], mode='bilinear', align_corners=False)
                        occ_mask = mask_builder(mono_disp, self.inputs['direct'] * (-1 if t_side == 's' else 1))
                        occ_mask = F.interpolate(occ_mask, [s * 2 for s in occ_mask.shape[2:]], mode='bilinear', align_corners=False)
                        occ_mask = (occ_mask > 0.8).to(torch.float)
                        
                    else:
                        mono_disp = outputs['mono_disp_0_{}'.format(t_side)].detach()
                        occ_mask = mask_builder(mono_disp, self.inputs['direct'] * (-1 if t_side == 's' else 1))
                        occ_mask = (occ_mask > 0.8).to(torch.float)

                    # compute fuse image
                    raw_img = self.inputs[target_name]
                    mono_img = m_projected_img
                    fused_mono = (occ_mask * edge_mask) * raw_img + (1 - (occ_mask * edge_mask)) * mono_img
                    outputs['fused_occ_color_{}'.format(t_side)] = fused_mono
                    outputs['mono_occ_mask_{}'.format(t_side)] = occ_mask
                    outputs['invalid_mask_{}'.format(t_side)] = 1 - (occ_mask * edge_mask)
                    projected_img = (1 - occ_mask) * raw_img + occ_mask * projected_img

                else:
                    raw_img = self.inputs[target_name]
                
                # with torch.no_grad():
                #     raw_feats = self.feat_net.get_feats(raw_img)
                # synth_feats = self.feat_net.get_feats(projected_img)
                # for feat_idx in range(3):
                #     rawf_name = 'stereo_raw_feats_{}_{}_{}'.format(
                #         id_frame, feat_idx, t_side)
                #     outputs[rawf_name] = raw_feats[feat_idx]
                #     synthf_name = 'stereo_synth_feats_{}_{}_{}'.format(
                #         id_frame, feat_idx, t_side)
                #     outputs[synthf_name] = synth_feats[feat_idx] 
        
        return outputs

    def _disp2depth(self, disp):
        """Convert network's sigmoid output into depth prediction."""
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth
    
    def get_parameters(self):
        if "Unified" in self.params_trained:
            # Unified-BB*0.25-CA*0.5-MO*1+DE*0.1
            bb_info, ca_info, mo_info = self.params_trained.split('-')[1:]
            def pdict_from_str(info):
                if info == '':
                    return {}
                sub_info = info.split('+')
                pdict = {}
                for s in sub_info:
                    name, lr = s.split('*')
                    pdict[name] = float(lr)
                return pdict
            g1_dict = pdict_from_str(bb_info)
            g2_dict = pdict_from_str(ca_info)
            g3_dict = pdict_from_str(mo_info)

            cross_module_block = [int(n) if len(n) == 1 else int(n[0]) for n in self.decoder_name.split('-')[1:]]
            midout_module_block = [int(n) if len(n) == 1 else int(n[0]) for n in self.decoder_name.split('_')[1:]]
            dec_num = len(self.decoder_ch_num)
            p_idx = -1
            cross_module_idxs = []
            midout_module_idxs = []
            dec_module_idxs = []
            # Dec blocks
            enc_feat_num = len(self.enc_ch_num) - 2
            for i in range(dec_num-1, -1, -1):
                p_idx += 1
                dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                if 'SDFA' in self.decoder_name and enc_feat_num >= 0:
                    p_idx += 1
                    dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                    p_idx += 1
                    dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                    p_idx += 1
                    dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                p_idx += 1
                dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                enc_feat_num -= 1

            # Out layer
            p_idx += 1
            dec_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))

            # Cross Attention Layers
            for _ in cross_module_block:
                p_idx += 1
                cross_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
            
                
            # Midout Layers
            for _ in midout_module_block:
                p_idx += 1
                midout_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
                if 'DeformConv' in self.decoder_name:
                    p_idx += 1
                    midout_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))
            
            # Midout Outlayer
            if 'DeformConv' in self.decoder_name or\
                'FinalBranch' in self.decoder_name:
                p_idx += 1
                midout_module_idxs.append(str(p_idx) + '.' if p_idx < 10 else str(p_idx))

            group_dict ={
                'CA': [],
                'MO': [],
                'BB': [],
                'DE': [],
            }
            # a = dict(self._networks.named_parameters())
            for k, v in self._networks.named_parameters():
                if v.requires_grad:
                    if k.startswith('1._convs.') and k[9:11] in cross_module_idxs:
                        group_dict['CA'] += [v]
                    elif k.startswith('1._convs.') and k[9:11] in midout_module_idxs:
                        group_dict['MO'] += [v]
                    else:
                        if k.startswith('1._convs.') and k[9:11] in dec_module_idxs:
                            group_dict['DE'] += [v]
                        group_dict['BB'] += [v]
            
            all_group = {}
            group1_list = [group_dict[k] for k in g1_dict.keys()]
            if group1_list:
                all_group['param_group1'] = ([{'params':group, 'lr': rate_lr} for group, rate_lr in zip(group1_list, g1_dict.values())], self.forward_st_epochs['param_group1'])
            group2_list = [group_dict[k] for k in g2_dict.keys()]
            if group2_list:
                all_group['param_group2'] = ([{'params':group, 'lr': rate_lr} for group, rate_lr in zip(group2_list, g2_dict.values())], self.forward_st_epochs['param_group2'])
            group3_list = [group_dict[k] for k in g3_dict.keys()]
            if group3_list:
                all_group['param_group3'] = ([{'params':group, 'lr': rate_lr} for group, rate_lr in zip(group3_list, g3_dict.values())], self.forward_st_epochs['param_group3'])
            
            return all_group
            
        else:
            return {'param_group': ([{'params': list(self.parameters())}], 0)}


class DepthProjector(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.height = image_size[0]
        self.width = image_size[1]

        meshgrid = np.meshgrid(range(self.width),
                               range(self.height),
                               indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(
            torch.stack(
                [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0),
            0)
        # self.pix_coords = self.pix_coords.repeat(1, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones],
                                                 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, T, K, img, is_mask=False):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(depth.shape[0], 1, -1) * cam_points
        cam_points = torch.cat(
            [cam_points, self.ones.repeat(depth.shape[0], 1, 1)], 1)

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, cam_points)

        pix_coords = cam_points[:, :2, :] \
            / (cam_points[:, 2, :].unsqueeze(1) + 1e-8)
        pix_coords = pix_coords.view(cam_points.shape[0], 2, self.height,
                                     self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        warped_img = F.grid_sample(img,
                                   pix_coords,
                                   mode='bilinear',
                                   padding_mode='border',
                                   align_corners=True)

        if is_mask:
            mask = ((pix_coords >= -1) & (pix_coords <= 1)).to(torch.float)
            mask = torch.min(mask, dim=3, keepdim=True)[0].permute(0, 3, 1, 2)
        else:
            mask = None

        return warped_img, mask

class DepthTransformer(object):
    def __init__(self, image_size, depth_range, device='cuda'):
        self.depth_range = depth_range
        self.ch_num = len(depth_range)

        self.height = image_size[0]
        self.width = image_size[1]
        meshgrid = np.meshgrid(range(self.width),
                               range(self.height),
                               indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)
        self.ones = torch.ones(1, 1, self.height * self.width).to(device)
        self.pix_coords = torch.unsqueeze(
            torch.stack(
                [self.id_coords[0].view(-1),
                self.id_coords[1].view(-1)], 0), 0).to(device)
        # self.pix_coords = self.pix_coords.repeat(1, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)

    def get_warped_volume(self, volume, inv_K, T, K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        P = torch.matmul(K, T)[:, :3, :]
        new_volume = []
        for ch_idx in range(self.ch_num):
            depth = self.depth_range[ch_idx]
            used_points = depth * cam_points
            used_points = torch.cat(
                [used_points, self.ones.repeat(used_points.shape[0], 1, 1)], 1)
            used_points = torch.matmul(P, used_points)
            pix_coords = used_points[:, :2, :] \
                / (used_points[:, 2, :].unsqueeze(1) + 1e-8)
            pix_coords = pix_coords.view(used_points.shape[0], 2, self.height,
                                        self.width)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            pix_coords[..., 0] /= self.width - 1
            pix_coords[..., 1] /= self.height - 1
            pix_coords = (pix_coords - 0.5) * 2
            # To adapt flip data augment
            warped_frame = F.grid_sample(volume[:, :, ch_idx, ...],
                                         pix_coords,
                                         mode='bilinear',
                                         padding_mode='zeros',
                                         align_corners=True)
            new_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(new_volume, dim=2)

    def get_warped_frame(self, x, inv_K, T, K):
        """Warp the images by disparity range with border padding."""
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        P = torch.matmul(K, T)[:, :3, :]
        frame_volume = []
        for ch_idx in range(self.ch_num):
            depth = self.depth_range[ch_idx]
            used_points = depth * cam_points
            used_points = torch.cat(
                [used_points, self.ones.repeat(used_points.shape[0], 1, 1)], 1)
            used_points = torch.matmul(P, used_points)
            pix_coords = used_points[:, :2, :] \
                / (used_points[:, 2, :].unsqueeze(1) + 1e-8)
            pix_coords = pix_coords.view(used_points.shape[0], 2, self.height,
                                        self.width)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            pix_coords[..., 0] /= self.width - 1
            pix_coords[..., 1] /= self.height - 1
            pix_coords = (pix_coords - 0.5) * 2
            # To adapt flip data augment
            warped_frame = F.grid_sample(x,
                                         pix_coords,
                                         mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))
        return torch.cat(frame_volume, dim=2)

class Feat_Net(object):
    def __init__(self, net_mode='vgg19', device='cpu'):
        self.feat_net = []
        if net_mode == 'vgg19':
            vgg = vgg19(pretrained=True, progress=False).features.to(device)
            vgg_feats = list(vgg.modules())
            vgg_layer_num = [5, 5, 9]
            read_module_num = 0
            for module_num in vgg_layer_num:
                self.feat_net.append(nn.Sequential())
                for _ in range(module_num):
                    self.feat_net[-1].add_module(
                        str(read_module_num), vgg_feats[read_module_num + 1])
                    read_module_num += 1
        else:
            raise NotImplementedError

    def get_feats(self, input_img):
        feats = []
        x = input_img
        for block_idx in range(len(self.feat_net)):
            x = self.feat_net[block_idx](x)
            feats.append(x)
        return feats

class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = nn.Parameter(torch.tensor(0), requires_grad=False)

class SelfOccluMask(nn.Module):
    def __init__(self, minDisp=0, maxDisp=21, device='cpu'):
        super(SelfOccluMask, self).__init__()
        self.minDisp = minDisp
        self.maxDisp = maxDisp
        self.device = device
        self.init_kernel()

    def init_kernel(self):
        self.convweights = torch.zeros(self.maxDisp - self.minDisp, 1, 3,
                                       self.maxDisp + 2).to(self.device)
        # self.occweights = torch.zeros(self.maxDisp, 1, 3,
        #                               self.maxDisp + 2).to(self.device)
        self.convbias = (torch.arange(self.minDisp, self.maxDisp).type(torch.FloatTensor) + 1 + self.minDisp).to(self.device)
        self.padding = nn.ReplicationPad2d((0, self.maxDisp + 1, 1, 1))
        for i in range(self.minDisp, self.maxDisp):
            self.convweights[i - self.minDisp, 0, :, 0:2] = 1 / 6
            self.convweights[i - self.minDisp, 0, :, i + 2:i + 3] = -1 / 3
            # self.occweights[i, 0, :, i + 2:i + 3] = 1 / 3

    def forward(self, dispmap, bsline):
        maskl = self.computeMask(dispmap, 'l')
        maskr = self.computeMask(dispmap, 'r')
        lind = bsline < 0
        rind = bsline > 0
        mask = torch.zeros_like(dispmap)
        mask[lind, :, :, :] = maskl[lind, :, :, :]
        mask[rind, :, :, :] = maskr[rind, :, :, :]
        return mask

    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            if direction == 'l':
                padmap = self.padding(dispmap)
                output = nn.functional.conv2d(padmap, self.convweights,
                                              self.convbias)
                output = torch.abs(output)
                mask, min_idx = torch.min(output, dim=1, keepdim=True)
                mask = mask.clamp(0, 1)
            elif direction == 'r':
                dispmap_opp = torch.flip(dispmap, dims=[3])
                padmap = self.padding(dispmap_opp)
                output = nn.functional.conv2d(padmap, self.convweights,
                                              self.convbias)
                output = torch.abs(output)
                mask, min_idx = torch.min(output, dim=1, keepdim=True)
                mask = mask.clamp(0, 1)
                mask = torch.flip(mask, dims=[3])
        return mask

class LapMask(nn.Module):
    def __init__(self, thres=0.13):
        super().__init__()
        self.thres = thres
        kernel =  []
        kernel = [[-1, -1, -1],
                  [-1, 8,  -1],
                  [-1, -1, -1]]
        
        self.kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
        
    
    def forward(self, depth):
        # left, right, top, bottom
        depth=depth.detach()
        self.kernel = self.kernel.to(depth)
        edge_mask = torch.abs(F.conv2d(depth / depth.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True),
                                       self.kernel, padding=1))
        edge_mask =  F.max_pool2d(edge_mask, 3, stride=1, padding=1)
        return edge_mask, 1 - edge_mask.clamp(0, self.thres) / self.thres
