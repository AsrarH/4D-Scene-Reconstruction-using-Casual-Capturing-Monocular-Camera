from builtins import breakpoint
from lib2to3.pgen2.driver import load_grammar
import warnings
import os.path as osp
warnings.filterwarnings("ignore")
import imageio 
import json
import os
import random
from torch.utils.data import Dataset
import math
import numpy as np
import torch
from PIL import Image

import copy
from typing import Optional, Tuple, Union

import numpy as np

from dataset.dycheck.utils import io, struct, types

from dataset import utils

def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + k3 * r))
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd
    d_r = k1 + r * (2.0 * k2 + 3.0 * k3 * r)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y
    return fx, fy, fx_x, fx_y, fy_x, fy_y

def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    x = xd.copy()
    y = yd.copy()
    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps,
            x_numerator / denominator,
            np.zeros_like(denominator),
        )
        step_y = np.where(
            np.abs(denominator) > eps,
            y_numerator / denominator,
            np.zeros_like(denominator),
        )
        x = x + step_x
        y = y + step_y
    return x, y

def points_to_local_points(
    points: np.ndarray,
    extrins: np.ndarray,
) -> np.ndarray:
    return utils.matv(extrins[..., :3, :3], points) + extrins[..., :3, 3]

def project(
    points: types.Array,
    intrins: types.Array,
    extrins: types.Array,
    radial_distortions: Optional[types.Array] = None,
    tangential_distortions: Optional[types.Array] = None,
    *,
    return_depth: bool = False,
    use_projective_depth: bool = True,
) -> types.Array:
    tensors_to_check = [intrins, extrins]
    if radial_distortions is not None:
        tensors_to_check.append(radial_distortions)
    if tangential_distortions is not None:
        tensors_to_check.append(tangential_distortions)
    isinstance(points, np.ndarray)
    assert all([isinstance(x, np.ndarray) for x in tensors_to_check])
    np_or_jnp = np
    local_points = points_to_local_points(points, extrins)
    normalized_pixels = np_or_jnp.where(
        local_points[..., -1:] != 0,
        local_points[..., :2] / local_points[..., -1:],
        0,
    )
    r2 = (normalized_pixels**2).sum(axis=-1, keepdims=True)
    if radial_distortions is not None:
        radial_scalars = 1 + r2 * (
            radial_distortions[..., 0:1]
            + r2
            * (
                radial_distortions[..., 1:2]
                + r2 * radial_distortions[..., 2:3]
            )
        )
    else:
        radial_scalars = 1
    if tangential_distortions is not None:
        tangential_deltas = 2 * tangential_distortions * np_or_jnp.prod(
            normalized_pixels,
            axis=-1,
            keepdims=True,
        ) + tangential_distortions[..., ::-1] * (
            r2 + 2 * normalized_pixels**2
        )
    else:
        tangential_deltas = 0
    normalized_pixels = normalized_pixels * radial_scalars + tangential_deltas
    pixels = utils.matv(
        intrins,
        np_or_jnp.concatenate(
            [
                normalized_pixels,
                np_or_jnp.ones_like(normalized_pixels[..., :1]),
            ],
            axis=-1,
        ),
    )[..., :2]
    if not return_depth:
        return pixels
    else:
        depths = (
            local_points[..., 2:]
            if use_projective_depth
            else np_or_jnp.linalg.norm(local_points, axis=-1, keepdims=True)
        )
        return pixels, depths

def load_json(filename, **kwargs):
    with open(filename) as f:
        return json.load(f, **kwargs)

def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()

def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()

blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()

def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
        c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
        c2w = (
            torch.Tensor(
                np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            ).to('cuda:1').float()
            @ c2w.to('cuda:1').float()
            @ blender2opencv.to('cuda:1').float()
        )
        return c2w

def normalize(v):
    return v / np.linalg.norm(v)

def average_poses(poses):
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_ = poses[..., 1].mean(0)
    x = normalize(np.cross(z[:3], y_[:3]))
    y = np.cross(x, z[:3])
    pose_avg = np.stack([x, y, z[:3], center[:3]], 1)
    return pose_avg

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=1007):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.zeros_like(render_poses[..., :1, :])], axis=1)
    render_poses[..., 3, 3] = 1
    render_poses = np.array(render_poses, dtype=np.float32)
    return render_poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses) , focal

class Camera(object):
    def __init__(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        focal_length: Union[np.ndarray, float],
        principal_point: np.ndarray,
        image_size: np.ndarray,
        skew: Union[np.ndarray, float] = 0.0,
        pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
        radial_distortion: Optional[np.ndarray] = None,
        tangential_distortion: Optional[np.ndarray] = None,
        *,
        use_center: bool = True,
        use_projective_depth: bool = True,
    ):
        if radial_distortion is None:
            radial_distortion = np.array([0, 0, 0], np.float32)
        if tangential_distortion is None:
            tangential_distortion = np.array([0, 0], np.float32)
        self.orientation = np.array(orientation, np.float32)
        self.position = np.array(position, np.float32)
        self.focal_length = np.array(focal_length, np.float32)
        self.principal_point = np.array(principal_point, np.float32)
        self.image_size = np.array(image_size, np.uint32)
        self.skew = np.array(skew, np.float32)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, np.float32)
        self.radial_distortion = np.array(radial_distortion, np.float32)
        self.tangential_distortion = np.array(tangential_distortion, np.float32)
        self.use_center = use_center
        self.use_projective_depth = use_projective_depth

    @classmethod
    def fromjson(cls, filename: types.PathType):
        camera_dict = load_json(filename)
        if "tangential" in camera_dict:
            camera_dict["tangential_distortion"] = camera_dict["tangential"]
        return cls(
            orientation=np.asarray(camera_dict["orientation"]),
            position=np.asarray(camera_dict["position"]),
            focal_length=camera_dict["focal_length"],
            principal_point=np.asarray(camera_dict["principal_point"]),
            image_size=np.asarray(camera_dict["image_size"]),
            skew=camera_dict["skew"],
            pixel_aspect_ratio=camera_dict["pixel_aspect_ratio"],
            radial_distortion=np.asarray(camera_dict["radial_distortion"]),
            tangential_distortion=np.asarray(camera_dict["tangential_distortion"]),
        )

    def asdict(self):
        return {
            "orientation": self.orientation,
            "position": self.position,
            "focal_length": self.focal_length,
            "principal_point": self.principal_point,
            "image_size": self.image_size,
            "skew": self.skew,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "radial_distortion": self.radial_distortion,
            "tangential_distortion": self.tangential_distortion,
        }

    @property
    def scale_factor_x(self):
        return self.focal_length

    @property
    def scale_factor_y(self):
        return self.focal_length * self.pixel_aspect_ratio

    @property
    def principal_point_x(self):
        return self.principal_point[0]

    @property
    def principal_point_y(self):
        return self.principal_point[1]

    @property
    def has_tangential_distortion(self):
        return any(self.tangential_distortion != 0)

    @property
    def has_radial_distortion(self):
        return any(self.radial_distortion != 0)

    @property
    def distortion(self):
        return np.concatenate(
            [
                self.radial_distortion[:2],
                self.tangential_distortion,
                self.radial_distortion[-1:],
            ]
        )

    @property
    def image_size_y(self):
        return self.image_size[1]

    @property
    def image_size_x(self):
        return self.image_size[0]

    @property
    def image_shape(self):
        return np.array([self.image_size_y, self.image_size_x], np.uint32)

    @property
    def optical_axis(self):
        return self.orientation[2, :]

    @property
    def up_axis(self):
        return -self.orientation[1, :]

    @property
    def translation(self):
        return -self.orientation @ self.position

    @property
    def intrin(self):
        return np.array(
            [
                [self.scale_factor_x, self.skew, self.principal_point_x],
                [0, self.scale_factor_y, self.principal_point_y],
                [0, 0, 1],
            ],
            np.float32,
        )

    @property
    def extrin(self):
        return np.concatenate(
            [
                np.concatenate(
                    [self.orientation, self.translation[..., None]], axis=-1
                ),
                np.array([[0, 0, 0, 1]], np.float32),
            ],
            axis=-2,
        )

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        y = (pixels[..., 1] - self.principal_point_y) / self.scale_factor_y
        x = (
            pixels[..., 0] - self.principal_point_x - y * self.skew
        ) / self.scale_factor_x
        if self.has_radial_distortion or self.has_tangential_distortion:
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1],
            )
        y = y * self.scale_factor_y + self.principal_point_y
        x = x * self.scale_factor_x + self.principal_point_x + y * self.skew
        return np.stack([x, y], axis=-1)

    def pixels_to_local_viewdirs(self, pixels: np.ndarray):
        y = (pixels[..., 1] - self.principal_point_y) / self.scale_factor_y
        x = (
            pixels[..., 0] - self.principal_point_x - y * self.skew
        ) / self.scale_factor_x
        if self.has_radial_distortion or self.has_tangential_distortion:
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1],
            )
        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    def pixels_to_viewdirs(self, pixels: np.ndarray) -> np.ndarray:
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")
        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))
        local_viewdirs = self.pixels_to_local_viewdirs(pixels)
        viewdirs = utils.matv(self.orientation.T, local_viewdirs)
        viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = viewdirs.reshape((*batch_shape, 3))
        return viewdirs

    def pixels_to_rays(self, pixels: np.ndarray) -> struct.Rays:
        viewdirs = self.pixels_to_viewdirs(pixels)
        return struct.Rays(
            origins=np.broadcast_to(self.position, viewdirs.shape),
            directions=viewdirs,
        )

    def pixels_to_cosa(self, pixels: np.ndarray) -> np.ndarray:
        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        return (rays_through_pixels @ self.optical_axis)[..., None]

    def pixels_to_points(
        self,
        pixels: np.ndarray,
        depth: np.ndarray,
        use_projective_depth: Optional[bool] = None,
    ) -> np.ndarray:
        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth
        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        cosa = 1 if not use_projective_depth else self.pixels_to_cosa(pixels)
        points = rays_through_pixels * depth / cosa + self.position
        return points

    def points_to_local_points(self, points: np.ndarray):
        return points_to_local_points(points, self.extrin)

    def project(
        self,
        points: np.ndarray,
        return_depth: bool = False,
        use_projective_depth: Optional[bool] = None,
    ):
        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth
        return project(
            points,
            self.intrin,
            self.extrin,
            self.radial_distortion,
            self.tangential_distortion,
            return_depth=return_depth,
            use_projective_depth=use_projective_depth,
        )

    def get_pixels(self, use_center: Optional[bool] = None):
        if use_center is None:
            use_center = self.use_center
        xx, yy = np.meshgrid(
            np.arange(self.image_size_x, dtype=np.float32),
            np.arange(self.image_size_y, dtype=np.float32),
        )
        offset = 0.5 if use_center else 0
        return np.stack([xx, yy], axis=-1) + offset

    def rescale(self, scale: float) -> "Camera":
        if scale <= 0:
            raise ValueError("scale needs to be positive.")
        camera = self.copy()
        camera.position *= scale
        return camera

    def translate(self, transl: np.ndarray) -> "Camera":
        camera = self.copy()
        camera.position += transl
        return camera

    def lookat(
        self,
        position: np.ndarray,
        lookat: np.ndarray,
        up: np.ndarray,
        eps: float = 1e-6,
    ) -> "Camera":
        look_at_camera = self.copy()
        optical_axis = lookat - position
        norm = np.linalg.norm(optical_axis)
        if norm < eps:
            raise ValueError(
                "The camera center and look at position are too close."
            )
        optical_axis /= norm
        right_vector = np.cross(optical_axis, up)
        norm = np.linalg.norm(right_vector)
        if norm < eps:
            raise ValueError("The up-vector is parallel to the optical axis.")
        right_vector /= norm
        camera_rotation = np.identity(3)
        camera_rotation[0, :] = right_vector
        camera_rotation[1, :] = np.cross(optical_axis, right_vector)
        camera_rotation[2, :] = optical_axis
        look_at_camera.position = position
        look_at_camera.orientation = camera_rotation
        return look_at_camera

    def undistort_image_domain(self) -> "Camera":
        camera = self.copy()
        camera.skew = 0
        camera.radial_distortion = np.zeros(3, dtype=np.float32)
        camera.tangential_distortion = np.zeros(2, dtype=np.float32)
        return camera

    def rescale_image_domain(self, scale: float) -> "Camera":
        if scale <= 0:
            raise ValueError("scale needs to be positive.")
        camera = self.copy()
        camera.focal_length *= scale
        camera.principal_point *= scale
        camera.image_size = np.array(
            (
                int(round(self.image_size[0] * scale)),
                int(round(self.image_size[1] * scale)),
            )
        )
        return camera

    def crop_image_domain(
        self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> "Camera":
        crop_left_top = np.array([left, top])
        crop_right_bottom = np.array([right, bottom])
        new_resolution = self.image_size - crop_left_top - crop_right_bottom
        new_principal_point = self.principal_point - crop_left_top
        if np.any(new_resolution <= 0):
            raise ValueError(
                "Crop would result in non-positive image dimensions."
            )
        new_camera = self.copy()
        new_camera.image_size = np.array(
            [int(new_resolution[0]), int(new_resolution[1])]
        )
        new_camera.principal_point = np.array(
            [new_principal_point[0], new_principal_point[1]]
        )
        return new_camera

    def copy(self) -> "Camera":
        return copy.deepcopy(self)

class Iphone_dataset(Dataset):
    def __init__(self, 
                 datadir,
                 split="train",
                 ratio=2.0,
                 use_bg_points=False,
                 cal_fine_bbox=False,
                 add_cam=False,
                 is_stack=False,
                 time_scale=1.0,
                 bbox=1.5,
                 N_random_pose=1007):
        self.img_wh = (int(720 / ratio), int(960 / ratio))
        
        self.white_bg = False
        self.time_scale = time_scale
        self.is_stack = is_stack
        self.N_random_pose = N_random_pose
        
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            self.meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        with open(f'{datadir}/extra.json', 'r') as f:
            extra_json = json.load(f)

        self.blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.center= np.array(scene_json["center"], dtype=np.float32)
        self.scale=scene_json['scale']

        self.near_far = [self.near, self.far]
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']
        self.bbox = np.array(extra_json['bbox'], dtype=np.float32)
        self.scene_bbox = torch.tensor(self.bbox)
        self._lookat = np.array(extra_json["lookat"], dtype=np.float32)
        self._up = np.array(extra_json["up"], dtype=np.float32)


        self.split = split
        self.all_img = dataset_json['ids']
        all_img_np = np.array(self.all_img)
        i_test = dataset_json['val_ids']
        i_test_np = np.array(i_test)
        test_ids = i_test_np.tolist()
        i_train = dataset_json['train_ids']
        i_train_np = np.array(i_train)
        train_ids = i_train_np.tolist()

        dataset_dict = io.load(osp.join("datasets/iphone/paper-windmill/dataset.json"))
        _frame_names = np.array(dataset_dict["ids"])

        metadata_dict = io.load(osp.join("datasets/iphone/paper-windmill/metadata.json"))
        self.time_ids = np.array(
            [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
        )
        time_min = self.time_ids.min()
        time_max = self.time_ids.max()
        self.normalized_time_ids = (self.time_ids - time_min) / (time_max - time_min)
        self.i_train = train_ids
        self.i_test = test_ids
        self.all_img_path = [f'{datadir}/rgb/2x/{i}.png' for i in all_img_np]

def uniq_time_ids(self):
        return np.unique(self.time_ids)

def extract_id_from_path(self,image_path):
        filename_with_extension = os.path.basename(image_path)
        id = os.path.splitext(filename_with_extension)[0]
        return id

def get_val_pose(self):
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

def translation(self,orientation,position):
        return -orientation @ position

def extrin(self,orientation,translation):
        return np.concatenate(
            [
                np.concatenate(
                    [orientation, translation[..., None]], axis=-1
                ),
                np.array([[0, 0, 0, 1]], np.float32),
            ],
            axis=-2,
        )

def extrin_to_c2w(self,orientation, translation):
        inv_rotation = orientation.T
        inv_translation = -np.dot(inv_rotation, translation)
        top_row = np.concatenate([inv_rotation, inv_translation[..., None]], axis=-1)
        bottom_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
        c2w_matrix = np.concatenate([top_row, bottom_row], axis=0)
        return c2w_matrix

def extract_dataset_information(self):
        device = 'cuda'  
        self.mat = []
        render_poses = []
        imgs = []
        all_imgs = []
        time = []
        poses = []
        for im in self.all_img_path:
            
            imgs.append(imageio.imread(im))
            image_id = self.extract_id_from_path(im)
            
            camera = (
                        Camera.fromjson(
                            osp.join(f"datasets/iphone/paper-windmill/camera/{image_id}.json")
                        )
                        .rescale_image_domain(1 / 2)
                        .translate(-self.center)
                        .rescale(self.scale)
                    )

         
            orientation = torch.tensor(camera.orientation, dtype=torch.float32)
            position = torch.tensor(camera.position, dtype=torch.float32)

            translation = self.translation(orientation,position)
            c2w = self.extrin_to_c2w(orientation.cpu(), translation.cpu())
            poses.append(c2w)

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        all_imgs.append(imgs)
        imgs = np.concatenate(all_imgs, 0)
        
        H,W ,_= imgs[0].shape
       
        H_tensor = torch.tensor(camera.image_size_x, dtype=torch.float32)
        W_tensor = torch.tensor(camera.image_size_y, dtype=torch.float32)
        
        H_tensor = H_tensor.to(device)
        W_tensor = W_tensor.to(device)
        
        self.focal = (
                0.5 * 720 / np.tan(0.5 * 179.49372779548082)
            )
        self.focal *= (
                self.img_wh[0] / 720
            )

        id_to_index_map = {img_id: index for index, img_id in enumerate(self.all_img)}
        self.i_train_indices = [id_to_index_map[img_id] for img_id in self.i_train]
        self.i_test_indices = [id_to_index_map[img_id] for img_id in self.i_test]

        return imgs, poses, self.normalized_time_ids, poses, self.normalized_time_ids, [camera.image_size_y, camera.image_size_x,camera.focal_length ], [self.i_train_indices,self.i_test_indices]
