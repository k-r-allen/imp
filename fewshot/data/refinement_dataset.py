import cv2
import numpy as np
import os
import gzip
import pickle as pkl

from fewshot.data.episode import Episode
from fewshot.data.data_factory import RegisterDataset
import pdb


class MetaDataset(object):

  def next_episode(self):
    """Get a new episode training."""
    pass


class RefinementMetaDataset(object):
  """A few-shot learning dataset with refinement (unlabeled) training. images.
  """

  def __init__(self, args, split, nway=5, nshot=1, num_unlabel=2, num_distractor=0, num_test=10, 
               label_ratio=1., mode_ratio=1., train_modes=True, cat_way=5., seed=0):
    """Creates a meta dataset.
    Args:
      args: arguments from command line input
      split: String.
      nway: Int. N way classification problem, default 5.
      nshot: Int. N-shot classification problem, default 1.
      num_unlabel: Int. Number of unlabeled examples per class, default 2.
      num_distractor: Int. Number of distractor classes, default 0.
      num_test: Int. Number of query images, default 10.
      label_ratio: ratio of labeled to unlabeled images (default 1.)
      mode_ratio: ratio of modes (sub-classes) to include in training
      train_modes: whether to use train set
      cat_way: N way classification over categories
      seed: Int. Random seed.
    """
    self._split = split
    self._cat_way = cat_way
    self._train_modes = train_modes
    self._nway = nway
    self._nshot = nshot
    self._num_unlabel = num_unlabel
    self._rnd = np.random.RandomState(seed)
    self._seed = seed
    self._mode_ratio = mode_ratio
    self._num_distractor = 0 if num_unlabel==0 else num_distractor

    self._num_test = num_test
    self._label_ratio = args.label_ratio if label_ratio is None else label_ratio

    self.read_dataset()

    # Build a set for quick query.
    self._label_split_idx = np.array(self._label_split_idx)
    self._rnd.shuffle(self._label_split_idx)
    self._label_split_idx_set = set(list(self._label_split_idx))
    self._unlabel_split_idx = list(filter(
        lambda _idx: _idx not in self._label_split_idx_set,
        range(self._labels.shape[0])))

    self._rnd.shuffle(self._unlabel_split_idx)

    self._unlabel_split_idx = np.array(self._unlabel_split_idx)
    if len(self._unlabel_split_idx) > 0:
      self._unlabel_split_idx_set = set(self._unlabel_split_idx)
    else:
      self._unlabel_split_idx_set = set()

    num_label_cls = len(self._label_str)
    self._num_classes = num_label_cls
    if hasattr(self, '_category_labels') and self._category_labels is not None:
      self.num_categories = len(np.unique(self._category_labels))
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)
    self._label_idict = {}
    self._category_nums = {}
    for cc in range(num_label_cls):
      self._label_idict[cc] = ex_ids[self._labels == cc]
    
    self.class_dict = {}
    for class_name in range(num_label_cls):
      ids = ex_ids[self._labels == class_name]
      # Split the image IDs into labeled and unlabeled.
      _label_ids = list(
          filter(lambda _id: _id in self._label_split_idx_set, ids))
      _unlabel_ids = list(
          filter(lambda _id: _id not in self._label_split_idx_set, ids))

      self.class_dict[class_name] = {
            'lbl': _label_ids,
            'unlbl': _unlabel_ids
        }

    self._nshot = nshot

  def mode_split(self):
    """Gets mode id splits.
    Returns:
      labeled_split: List of int.
    """
    print('Label split using seed {:d}'.format(self._seed))
    rnd = np.random.RandomState(self._seed)
    num_cats = len(np.unique(self._category_labels))

    mode_split = []
    self.coarse_labels = [[] for _ in range(num_cats)]
    for sub, sup in enumerate(self._category_labels):
      self.coarse_labels[sup].append(sub)
    for sup in range(0, len(self.coarse_labels)):
      mode_split.extend(list(np.random.choice(self.coarse_labels[sup],max(1,int(self._mode_ratio*len(self.coarse_labels[sup]))),replace=False)))
    print("Mode split {}".format(len(mode_split)))
    return sorted(mode_split)

  def read_mode_split(self):
    cache_path_modesplit = self.get_mode_split_path()
    if os.path.exists(cache_path_modesplit):
      self._class_train_set = np.loadtxt(cache_path_modesplit, dtype=np.int64)
    else:
      if self._split in ['train', 'trainval']:
        print('Use {}% image for mode split.'.format(
            int(self._mode_ratio * 100)))
        self._class_train_set = self.mode_split()
      elif self._split in ['val', 'test']:
        print('Use all image in mode split, since we are in val/test')
        self._class_train_set = np.arange(self._images.shape[0])
      else:
        raise ValueError('Unknown split {}'.format(self._split))
      self._class_train_set = np.array(self.mode_split(), dtype=np.int64)
      self.save_mode_split()

  def save_mode_split(self):
    np.savetxt(self.get_mode_split_path(), self._class_train_set, fmt='%d')

  def get_mode_split_path(self):
    mode_ratio_str = '_' + str(int(self._mode_ratio * 100))
    seed_id_str = '_' + str(self._seed)
    if self._split in ['train', 'trainval']:
      cache_path = os.path.join(
          self._folder, self._split + '_modesplit' +
          mode_ratio_str + seed_id_str + '.txt')
    elif self._split in ['val', 'test']:
      cache_path = os.path.join(
          self._folder,
          self._split + '_modesplit' + '.txt')
    return cache_path


  def process_category_labels(self, labels):
    i = 0
    mydict = {}
    if isinstance(labels[0], basestring):
      for item in labels:
        if '/' in item:
          item = item.split('/')[0]
        if(i>0 and item in mydict):
          continue
        else:    
           mydict[item] = i
           i = i+1

      k=[]
      for item in labels:
        if '/' in item:
          item = item.split('/')[0]
          k.append(mydict[item])
      return k
    else:
      return list(labels)

  def episodic_labels(self, labels):
    i = 0
    mydict = {}
    for item in labels:
      if(i>0 and item in mydict):
        continue
      else:    
         mydict[item] = i
         i = i+1

    k=[]
    for item in labels:
        k.append(mydict[item])
    return k  

  def shuffle_labels(self):
    # Build a set for quick query.
    for cc in self._label_idict.keys():
      self._rnd.shuffle(self._label_idict[cc])


  def read_dataset(self):
    """Reads data from folder or cache."""
    raise NotImplemented()

  def label_split(self):
    """Gets label/unlabel image splits.
    Returns:
      labeled_split: List of int.
    """
    print('Label split using seed {:d}'.format(self._seed))
    rnd = np.random.RandomState(self._seed)
    num_label_cls = len(self._label_str)
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)

    labeled_split = []
    for cc in range(num_label_cls):
      cids = ex_ids[self._labels == cc]
      rnd.shuffle(cids)
      labeled_split.extend(cids[:int(len(cids) * self._label_ratio)])
    print("Total number of classes {}".format(num_label_cls))
    print("Labeled split {}".format(len(labeled_split)))
    print("Total image {}".format(num_ex))
    return sorted(labeled_split)

  def filter_classes(self, class_seq):
    idxs = list(
          filter(lambda _id: self._label_general[_id] in self.okay_classes[self._split], class_seq))
    return idxs

  def next_episode(self, within_category=False):
    """Gets a new episode.
    within_category: bool. Whether or not to choose classes
    which all belong to the same more general category.
    (Only applicable for datasets with self._category_labels defined).
    """

    num_label_cls = len(self._label_str)

    if self._mode_ratio < 1.0:
      if self._train_modes:
        self.class_seq = list(
              filter(lambda _id: _id in self._class_train_set, range(0, num_label_cls)))
      else:
        self.class_seq = list(
          filter(lambda _id: _id not in self._class_train_set, range(0, num_label_cls)))
    else:
      self.class_seq = np.arange(num_label_cls)
      
    train_img_ids = []
    train_labels = []
    test_img_ids = []
    test_labels = []

    train_unlabel_img_ids = []
    non_distractor = []

    train_labels_str = []
    test_labels_str = []

    self._rnd.shuffle(self.class_seq)

    ##Get a list of image indices (class_seq_i) which are within cat_way (number of categories per episode) randomly selected categories
    if within_category and self._cat_way != -1:
      assert hasattr(self, "_category_labels")
      cat_labels = np.unique(self._category_labels)
      num_cats = len(cat_labels)

      cat_idxs = self._rnd.choice(cat_labels, min(self._cat_way, num_cats),replace=False)
      allowable_inds = np.empty((1))
      for cat_idx in cat_idxs:
        current_inds = np.where(np.array(self._category_labels) == cat_idx)[0]
        filtered_inds = list(filter(lambda _id: _id in self.class_seq, current_inds))
        self._rnd.shuffle(filtered_inds)
        allowable_inds = np.concatenate((allowable_inds, filtered_inds[0:min(self._nway,len(filtered_inds))]))
      class_seq_i = (allowable_inds[1:]).astype(np.int64)
      self._rnd.shuffle(class_seq_i)
      total_way = len(class_seq_i)

    else:
      total_way = self._nway
      class_seq_i = self.class_seq

    is_training = self._split in ["train", "trainval"]
    assert is_training or self._split in ["val", "test"]

    for ii in range(total_way + self._num_distractor):
      cc = class_seq_i[ii]

      _ids = self._label_idict[cc]

      # Split the image IDs into labeled and unlabeled.
      _label_ids = list(
          filter(lambda _id: _id in self._label_split_idx_set, _ids))
      _unlabel_ids = list(
          filter(lambda _id: _id not in self._label_split_idx_set, _ids))

      self._rnd.shuffle(_label_ids)
      self._rnd.shuffle(_unlabel_ids)
      if not is_training:
        train_idx = self._nshot+self._num_unlabel
      else:
        train_idx = self._nshot
        
      _label_train_ids = _label_ids[:train_idx]
      _label_test_ids = _label_ids[train_idx:]  
      self._rnd.shuffle(_label_train_ids)
      self._rnd.shuffle(_label_test_ids)

      test_end_idx = self._nshot


      class_idx = [cc, ii]

      if self._num_test == -1:
        if is_training:
          num_test = len(_label_test_ids)
        else:
          num_test = len(_label_test_ids) - self._num_unlabel - 1
      else:
        num_test = self._num_test
        if is_training:
          assert num_test <= len(_label_test_ids)
        else:
          assert num_test <= len(_label_test_ids) - self._num_unlabel

      # Add support set and query set (not for distractors).
      if hasattr(self, "_category_labels") and self._category_labels is not None:
        label_strs = self._category_labels
      else:
        label_strs = self._label_str

      if ii < total_way:
        train_img_ids.extend(_label_train_ids[:self._nshot])

        # Use the rest of the labeled image as queries, if num_test = -1.
        QUERY_SIZE_LARGE_ERR_MSG = (
            "Query + reference should be less than labeled examples." +
            "Num labeled {} Num test {} Num shot {}".format(
                len(_label_ids), self._num_test, self._nshot))
        assert self._nshot + self._num_test <= len(
            _label_ids), QUERY_SIZE_LARGE_ERR_MSG

        test_img_ids.extend(_label_test_ids[:num_test])

        train_labels.extend([class_idx] * self._nshot)
        train_labels_str.extend([label_strs[cc]] * self._nshot)
        test_labels.extend([class_idx] * num_test)
        test_labels_str.extend([label_strs[cc]] * num_test)
        non_distractor.extend([class_idx] * self._num_unlabel)
      else:
        non_distractor.extend([[-1,-1]]* self._num_unlabel)


      # Add unlabeled images here.
      if is_training:
        # Use labeled, unlabeled split here for refinement.
        train_unlabel_img_ids.extend(_unlabel_ids[:self._num_unlabel])

      else:
        train_unlabel_img_ids.extend(_label_train_ids[
            self._nshot:self._nshot + self._num_unlabel])

    train_img = self.get_images(train_img_ids) / 255.0
    train_unlabel_img = self.get_images(train_unlabel_img_ids) / 255.0
    test_img = self.get_images(test_img_ids) / 255.0
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    if hasattr(self, "_category_labels"):
      train_labels_str = np.hstack((np.array(train_labels_str)[:,None], np.array(self.episodic_labels(train_labels_str))[:,None]))
      test_labels_str = np.hstack((np.array(test_labels_str)[:,None], np.array(self.episodic_labels(test_labels_str))[:,None]))
    else:
      train_labels_str = np.array(train_labels_str)
      test_labels_str = np.array(test_labels_str)

    non_distractor = np.array(non_distractor)

    test_ids_set = set(test_img_ids)
    for _id in train_unlabel_img_ids:
      assert _id not in test_ids_set



    return Episode(
        x_train=train_img,
        train_indices = train_img_ids,
        y_train=train_labels,
        x_test=test_img,
        test_indices=test_img_ids,
        y_test=test_labels,
        x_unlabel=train_unlabel_img,
        y_unlabel=non_distractor,
        unlabel_indices=train_unlabel_img_ids,
        y_train_str=train_labels_str,
        y_test_str=test_labels_str)

  def reset(self):
    self._rnd = np.random.RandomState(self._seed)

  @property
  def num_classes(self):
    return self._num_classes
