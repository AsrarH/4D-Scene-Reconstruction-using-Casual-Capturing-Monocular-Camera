import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class DirectTemporalNeRF(nn.Module):
    def __init__(self, D=8, W=9, input_ch=3, input_ch_views=1, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True):
        super(DirectTemporalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self.app_dim = 84
        self.dummy_plane = nn.Parameter(torch.randn(3, 100, 100))  
        self.dummy_line_time = nn.Parameter(torch.randn(3, 100, 100))  
        self.dummy_basis_mat = nn.Linear(300, self.app_dim)  
        self.matMode = [[0, 3], [1, 3], [2, 3]]  # X-T, Y-T, Z-T
        self.vecMode = [3, 3, 3]  # All time
        app_n_comp: Union[int, List[int]] = [48, 48, 48]
        self.app_n_comp = app_n_comp
        gridSize = [64, 64, 64,64]
        init_shift: float = 0.0
        self.init_shift = init_shift
        self.time_grid = 16
        self.align_corners=True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.init_scale = float = 0.1
        self._occ = NeRFOriginal(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views,
                                 input_ch_time=input_ch_time, output_ch=output_ch, skips=skips,
                                 use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
        
        self._time, self._time_out = self.create_time_net()
        
        self.app_plane = self.init_one_triplane(
                self.app_n_comp, self.gridSize, self.device
            )
        self.app_basis_mat = torch.nn.Linear(
                                sum(self.app_n_comp)*21, self.app_dim, bias=False
                            ).to(self.device)

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def init_one_triplane(self, n_component, gridSize, device):
            plane_coef, line_time_coef = [], []

            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]

                plane_coef.append(
                    torch.nn.Parameter(
                        self.init_scale
                        * torch.randn(
                            (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                        )
                        + self.init_shift
                    )
                )


            return torch.nn.ParameterList(plane_coef).to(device)
      # plane_coef= []
      # for i in range(3):  # X-T, Y-T, Z-T
      #     print(i)
      #     # Ensure gridSize[i] is the correct spatial dimension
      #     # time_dimension_size is the temporal dimension
      #     spatial_dimension = gridSize[i]
      #     print("spatial_dimension",spatial_dimension)
      #     time_dimension_size = self.time_grid

      #     plane_coef.append(
      #         torch.nn.Parameter(
      #             self.init_scale
      #             * torch.randn((1, n_component[i], spatial_dimension, time_dimension_size))
      #             + self.init_shift
      #         )
      #     )


      # return torch.nn.ParameterList(plane_coef).to(device)

 

    def compute_appfeature(self, xyz_sampled, frame_time):
        xyz_sampled_reshaped = xyz_sampled.view(-1, 21, 3)  # Reshaping for processing
        plane_feat = []
        B = xyz_sampled.shape[0]  # Batch size 
        for i in range(frame_time.shape[1]):  # Loop through time instances
            xyz_at_time_i = xyz_sampled_reshaped[:, i, :3]  # Extract spatial coordinates
            time_at_i = frame_time[:, i].unsqueeze(1)  # Extract time coordinates

            # Combine spatial and temporal coordinates for each plane
            plane_coord_xt = torch.cat((xyz_at_time_i[:, 0:1], time_at_i), dim=-1)
            plane_coord_yt = torch.cat((xyz_at_time_i[:, 1:2], time_at_i), dim=-1)
            plane_coord_zt = torch.cat((xyz_at_time_i[:, 2:3], time_at_i), dim=-1)

            # # Reshape for grid_sample
            plane_coord_xt = plane_coord_xt.unsqueeze(1)
            plane_coord_yt = plane_coord_yt.unsqueeze(1)
            plane_coord_zt = plane_coord_zt.unsqueeze(1)
            plane_coord = (
            torch.stack(
                (
                    plane_coord_xt,
                    plane_coord_yt,
                    plane_coord_zt,
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
            # plane_coord_xt = plane_coord_xt.expand(B, -1, -1, -1)
            # plane_coord_yt = plane_coord_yt.expand(B, -1, -1, -1)
            # plane_coord_zt = plane_coord_zt.expand(B, -1, -1, -1)

            # print("Grid tensor xt size:", plane_coord_xt.size())
            # print("Grid tensor yt size:", plane_coord_yt.size())
            # print("Grid tensor zt size:", plane_coord_zt.size())
            # Grid sampling for each X-T, Y-T, Z-T plane
            for idx_plane in range(len(self.app_plane)):
                # print("self.app_plane[idx_plane]",self.app_plane[idx_plane].shape)
                # print("plane_coord[[idx_plane]]",plane_coord[[idx_plane]].shape)
                plane_feat.append(
                    F.grid_sample(
                        self.app_plane[idx_plane],
                        plane_coord[[idx_plane]],
                        align_corners=self.align_corners,
                    ).view(-1, *xyz_sampled.shape[:1])
                )
                # plane_feat.append(
                #     F.grid_sample(
                #         self.app_plane[idx_plane],
                #         plane_coord_xt if idx_plane == 0 else 
                #         plane_coord_yt if idx_plane == 1 else 
                #         plane_coord_zt,
                #         align_corners=self.align_corners,
                #     ).view(-1, *xyz_sampled.shape[:1])
                # )


        # Processing and combining features
        plane_feat = torch.cat(plane_feat)


        return self.app_basis_mat(plane_feat.T)





    def query_time(self, new_pts, t, net, net_final):
        # Compute appearance features using compute_appfeature
        self.app_features = self.compute_appfeature(new_pts, t)
        # Prepare input for the network
        h = self.app_features
        for i, l in enumerate(net):
            h = net[i](h) # first 27,256,283 ..
            h = F.relu(h)
            if i in self.skips:
                # Concatenate original input if needed at skip connections
                h = torch.cat([new_pts, h], -1)

        return net_final(h)


    def forward(self, x, ts):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = ts[0]

        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = t[0, 0]
        if cur_time == 0. and self.zero_canonical:
            dx = torch.zeros_like(input_pts[:, :3])
        else:
            dx = self.query_time(input_pts, t, self._time, self._time_out)
            input_pts_orig = input_pts[:, :3]
            input_pts = self.embed_fn(input_pts_orig + dx)
        out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        return out, dx


class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model

class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] +
        #     [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def hsv_to_rgb(h, s, v):
    '''
    h,s,v in range [0,1]
    '''
    hi = torch.floor(h * 6)
    f = h * 6. - hi
    p = v * (1. - s)
    q = v * (1. - f * s)
    t = v * (1. - (1. - f) * s)

    rgb = torch.cat([hi, hi, hi], -1) % 6
    rgb[rgb == 0] = torch.cat((v, t, p), -1)[rgb == 0]
    rgb[rgb == 1] = torch.cat((q, v, p), -1)[rgb == 1]
    rgb[rgb == 2] = torch.cat((p, v, t), -1)[rgb == 2]
    rgb[rgb == 3] = torch.cat((p, q, v), -1)[rgb == 3]
    rgb[rgb == 4] = torch.cat((t, p, v), -1)[rgb == 4]
    rgb[rgb == 5] = torch.cat((v, p, q), -1)[rgb == 5]
    return rgb


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d



def custom_searchsorted(cdf, u, side='right'):
    """
    A custom implementation of searchsorted for 2D tensors.

    Args:
        cdf (torch.Tensor): 2D tensor where each row is a CDF.
        u (torch.Tensor): 2D tensor of values to search for in each row of cdf.
        side (str): If 'left', the index of the first suitable location found is given. 
                    If 'right', return the last such index. If there is no suitable index, 
                    return either 0 or the size of cdf.

    Returns:
        torch.Tensor: 2D tensor of indices.
    """
    assert cdf.ndim == 2 and u.ndim == 2, "cdf and u must be 2D tensors"

    # Assuming cdf and u have the same number of rows
    num_rows = cdf.shape[0]
    indices = torch.zeros(u.shape, dtype=torch.long, device=cdf.device)

    for i in range(num_rows):
        expanded_cdf = cdf[i].unsqueeze(0).expand(u.shape[1], -1)
        expanded_u = u[i].unsqueeze(-1)
        if side == 'right':
            indices[i] = torch.sum(expanded_cdf <= expanded_u, dim=1)
        elif side == 'left':
            indices[i] = torch.sum(expanded_cdf < expanded_u, dim=1)
        else:
            raise ValueError("side must be either 'left' or 'right'")

    return indices

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = custom_searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
