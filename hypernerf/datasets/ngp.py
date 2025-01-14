# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Casual Volumetric Capture datasets.

Note: Please benchmark before submitted changes to this module. It's very easy
to introduce data loading bottlenecks!
"""
import json
from typing import List, Tuple

from absl import logging
import cv2
import gin
import numpy as np

from hypernerf import camera as cam
from hypernerf import gpath
from hypernerf import types
from hypernerf import utils
from hypernerf.datasets import core

FLIP_MAT = np.array([
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
])

def load_scene_info(
    data_dir: types.PathType) -> Tuple[np.ndarray, float, float, float]:
  """Loads the scene center, scale, near and far from scene.json.

  Args:
    data_dir: the path to the dataset.

  Returns:
    scene_center: the center of the scene (unscaled coordinates).
    scene_scale: the scale of the scene.
    near: the near plane of the scene (scaled coordinates).
    far: the far plane of the scene (scaled coordinates).
  """
  # TODO: define these in transforms.json?
  #scene_json_path = gpath.GPath(data_dir, 'scene.json')
  #with scene_json_path.open('r') as f:
  #  scene_json = json.load(f)
  scene_scale = 1.0
  
  scene_center = [
    0.0,
    0.0,
    0.0
  ]

  near = 0.01
  far = 16

  return scene_center, scene_scale, near, far


def _load_image(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  with path.open('rb') as f:
    raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    image = np.asarray(image).astype(np.float32) / 255.0
    return image


def _load_dataset_ids(transforms_path: types.PathType) -> Tuple[List[str], List[str]]:
  """Loads dataset IDs."""
  logging.info('*** Loading dataset IDs from %s', transforms_path)
  data: dict
  with transforms_path.open('r') as f:
    data = json.load(f)
  
  num_frames = len(data['frames'])
  train_skip = 10

  train_ids = [str(i) for i in range(num_frames) if i % train_skip != 0]
  val_ids = [str(i) for i in range(num_frames) if i % train_skip == 0]

  return train_ids, val_ids


@gin.configurable
class NGPDataSource(core.DataSource):
  """Data loader for videos."""

  def __init__(self,
               data_dir: str = gin.REQUIRED,
               image_scale: int = gin.REQUIRED,
               shuffle_pixels: bool = False,
               transforms_path: str = None,
               **kwargs):
    self.data_dir = gpath.GPath(data_dir)
    
    if transforms_path == None:
      self.transforms_path = gpath.GPath(data_dir, 'transforms.json')
    else:
      self.transforms_path = gpath.GPath(transforms_path) 

    if not self.transforms_path.exists():
      logging.warning(f'transforms.json does not exist: {str(self.transforms_path)}', )
      return []

    # Load IDs from JSON if it exists. This is useful since COLMAP fails on
    # some images so this gives us the ability to skip invalid images.
    train_ids, val_ids = _load_dataset_ids(self.transforms_path)
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)
    self.scene_center, self.scene_scale, self._near, self._far = (
        load_scene_info(self.data_dir))

    self.image_scale = image_scale
    self.shuffle_pixels = shuffle_pixels

    self.rgb_dir = gpath.GPath(data_dir, 'this_is_hypernerf', 'rgb', f'{image_scale}x')
    self.depth_dir = gpath.GPath(data_dir, 'this_is_hypernerf', 'depth', f'{image_scale}x')
    self.camera_dir = gpath.GPath(data_dir, 'this_is_hypernerf', 'camera')

    self.checkpoint_dir = gpath.GPath(data_dir, 'checkpoints', 'hypernerf')
    
    with open(self.transforms_path, 'r') as f:
      self.transforms = json.load(f)

  @property
  def near(self) -> float:
    return self._near

  @property
  def far(self) -> float:
    return self._far

  def get_rgb_path(self, item_id: str) -> types.PathType:
    return self.rgb_dir / f'{item_id}.png'

  def load_rgb(self, item_id: str) -> np.ndarray:
    return _load_image(self.data_dir / self.transforms['frames'][int(item_id)]['file_path'])

  def load_camera(self,
                  item_id: str,
                  scale_factor = 1.0,
                  transforms: dict = None) -> cam.Camera:
    data = transforms
    if transforms == None:
      data = self.transforms
    
    frame = data['frames'][int(item_id)]

    rot_mat = np.array(frame['hypernerf']['orientation'])
    pos_vec = np.array(frame['hypernerf']['translation'])
    camera = cam.Camera(
      orientation=rot_mat,
      position=pos_vec,
      focal_length=data['fl_x'],
      principal_point=np.array([scale_factor * data['cx'], scale_factor * data['cy']]),
      image_size=np.array([scale_factor * data['w'], scale_factor * data['h']]),
      skew=0.0,
      pixel_aspect_ratio=1.0,
      radial_distortion=[data['k1'], data['k2'], 0],
      tangential_distortion=[data['p1'], data['p2']],
      dtype=np.float32
    )

    return camera

  def glob_cameras(self, path):
    path = gpath.GPath(path)
    return sorted(path.glob(f'*{self.camera_ext}'))

  def load_test_cameras(self, count=None, transforms_path:str=None):
    data = self.transforms
    if transforms_path != None:
      with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    cameras = [self.load_camera(i, transforms=data) for i in range(len(data['frames']))]

    return cameras

  def load_points(self, shuffle=False):
    with (self.data_dir / 'points.npy').open('rb') as f:
      points = np.load(f)
    points = (points - self.scene_center) * self.scene_scale
    points = points.astype(np.float32)
    if shuffle:
      logging.info('Shuffling points.')
      shuffled_inds = self.rng.permutation(len(points))
      points = points[shuffled_inds]
    logging.info('Loaded %d points.', len(points))
    return points

  def get_appearance_id(self, item_id):
    return int(item_id) # self.metadata_dict[item_id]['appearance_id']

  def get_camera_id(self, item_id):
    return 0 # self.metadata_dict[item_id]['camera_id']

  def get_warp_id(self, item_id):
    return int(item_id) # self.metadata_dict[item_id]['warp_id']

  def get_time_id(self, item_id):
    return int(item_id)
    if 'time_id' in self.metadata_dict[item_id]:
      return self.metadata_dict[item_id]['time_id']
    else:
      # Fallback for older datasets.
      return self.metadata_dict[item_id]['warp_id']
