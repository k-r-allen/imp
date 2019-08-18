import csv
import cv2
import numpy as np
import os
import pickle as pkl
import scipy
import h5py
from scipy.misc import imread, imresize

from collections import namedtuple
from tqdm import tqdm

from fewshot.data.data_factory import RegisterDataset
from fewshot.data.episode import Episode
from fewshot.data.refinement_dataset import RefinementMetaDataset
import pdb
AL_Instance = namedtuple('AL_Instance',
                         'n_class, n_distractor, k_train, k_test, k_unlbl')
DatasetT = namedtuple('Dataset', 'data, labels, indices')

# TODO: put in config file
N_IMAGES = 600
N_INPUT = 84
IMAGES_PATH = "images/"
CSV_FILES = {
    'train': 'fewshot/data/mini_imagenet_split/Ravi/train.csv',
    'val': 'fewshot/data/mini_imagenet_split/Ravi/test.csv',
    'test': 'fewshot/data/mini_imagenet_split/Ravi/val.csv'
}

# Fixed random seed to get same split of labeled vs unlabeled items for each class
FIXED_SEED = 22


@RegisterDataset("mini-imagenet")
class MiniImageNetDataset(RefinementMetaDataset):

  def __init__(self,
               args,
               folder,
               split,
               nway=5,
               nshot=1,
               num_unlabel=5,
               num_distractor=5,
               num_test=15,
               split_def="",
               label_ratio=0.4,
               aug_90=False,
               mode_ratio=1.,
               train_modes=True,
               cat_way=-1,
               seed=FIXED_SEED):

    self._folder = folder
    self._split = split
    self.images_path = os.path.join(self._folder, 'images/')
    self.name = 'mini-imagenet'
    self.n_input = 84
    if args.disable_distractor:
      num_distractor = 0
    super(MiniImageNetDataset,
          self).__init__(args, split, nway, nshot, num_unlabel, num_distractor,
                         num_test, label_ratio, mode_ratio, train_modes, cat_way, seed)


  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self._folder,
                              "mini-imagenet-cache-" + self._split + ".hdf5")

    return cache_path


  def read_cache(self):
    """Reads dataset from cached pklz file."""
    cache_path = self.get_cache_path()
    if os.path.exists(cache_path):
      data = h5py.File(cache_path,'r')
      self._images = np.array(data.get('images'))
      self._labels = np.array(data.get('labels'))
      self._label_str = np.array(data.get('label_str'))
      self._category_labels = None
      self.read_label_split()
      return True
    else:
      return False

  def read_dataset(self):
    # Read data from folder or cache.
    if not self.read_cache():
      self._save_cache(self._split, CSV_FILES[self._split])
      self.read_label_split()

  def read_label_split(self):
    cache_path_labelsplit = self.get_label_split_path()
    if os.path.exists(cache_path_labelsplit):
      self._label_split_idx = np.loadtxt(cache_path_labelsplit, dtype=np.int64)
    else:
      if self._split in ['train', 'trainval']:
        print('Use {}% image for labeled split.'.format(
            int(self._label_ratio * 100)))
        self._label_split_idx = self.label_split()
      elif self._split in ['val', 'test']:
        print('Use all image in labeled split, since we are in val/test')
        self._label_split_idx = np.arange(self._images.shape[0])
      else:
        raise ValueError('Unknown split {}'.format(self._split))
      self._label_split_idx = np.array(self.label_split(), dtype=np.int64)
      self.save_label_split()

  def _save_cache(self, split, csv_filename):
    cache_path = self.get_cache_path()
    img_data = []
    label_str = []
    labels = []

    class_dict = {}
    i = 0
    with open(csv_filename) as csv_file:
      csv_reader = csv.reader(csv_file)
      for (image_filename, class_name) in csv_reader:
        if 'label' not in class_name:
          img = imresize(
                  imread(self.images_path + image_filename), (self.n_input,
                                                              self.n_input, 3))
          #img = np.rollaxis(img, 2)

          img_data.append(img)          
          if class_name not in label_str:
            label_str.append(class_name)
          labels.append(label_str.index(class_name))

          i += 1

    self.img_data = np.stack(img_data)
    self._images = self.img_data
    self._labels = np.array(labels)
    self._label_str = np.array(label_str)
    h_file = h5py.File(self.get_cache_path())                                                                                                                                                                                
    h_file.create_dataset('images',data=self.img_data)
    h_file.create_dataset('labels',data=labels)
    h_file.create_dataset('label_str',data=label_str)
    h_file.close()

  def get_label_split_path(self):
  	   splitfile = os.path.join(self._folder, "mini-imagenet-labelsplit-" +
                       self._split + "-{}.pkl".format(self._seed))
  	   return splitfile


  def save_label_split(self):
    np.savetxt(self.get_label_split_path(), self._label_split_idx, fmt='%d')


  def reset(self):
    self._rnd = np.random.RandomState(self._seed)

  def get_images(self, inds):
    return self._images[inds]

  @property
  def num_classes(self):
    return self._num_classes



