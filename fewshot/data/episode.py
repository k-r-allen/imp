import pdb
import numpy as np
import torch
class Episode(object):

  def __init__(self,
               x_train,
               y_train,
               train_indices,
               x_test,
               y_test,
               test_indices,
               x_unlabel=None,
               y_unlabel=None,
               unlabel_indices=None,
               y_train_str=None,
               y_test_str=None):
    """Creates a miniImageNet episode.
    Args:
      x_train:  [N, ...]. Training data.
      y_train: [N]. Training label.
      x_test: [N, ...]. Testing data.
      y_test: [N]. Testing label.
    """
    self._x_train = x_train
    self._train_indices = train_indices
    self._y_train = y_train
    self._x_test = x_test
    self._y_test = y_test
    self._test_indices = test_indices
    self._x_unlabel = x_unlabel
    self._y_unlabel = y_unlabel
    self._unlabel_indices = unlabel_indices
    self._y_train_str = y_train_str
    self._y_test_str = y_test_str

  def next_batch(self):
    return self

  def next_batch_separate(self, classes, total_num_classes):
    example_range, y_train_labs = np.where(self._y_train[:,1][:,None] == classes)
    x_train = self._x_train[example_range,:,:,:]
    y_train = self._y_train[example_range,:]
    y_train[:,1] = y_train_labs
    train_indices = (np.array(self._train_indices)[example_range]).tolist()

    test_idxs, y_test_labs = np.where(self._y_test[:,1][:,None] == classes)
    x_test = self._x_test[test_idxs,:,:,:]
    y_test = self._y_test[test_idxs,:]
    y_test[:,1] = y_test_labs
    test_indices = (np.array(self._test_indices)[test_idxs]).tolist()

    if self._y_unlabel is not None and len(self._y_unlabel) > 0:
      unlabel_indices, unlabel_episode = np.where(self._y_unlabel[:,1][:,None] == np.array(classes.tolist() + range(total_num_classes, total_num_classes + 1+len(classes))))
      x_unlabel = self._x_unlabel[unlabel_indices,:,:,:]
      y_unlabel = self._y_unlabel[unlabel_indices,:]
      unlabel_img_indices = (np.array(self._unlabel_indices)[unlabel_indices]).tolist()
    else:
      x_unlabel = None
      y_unlabel = None
      unlabel_img_indices = None

    if self._y_train_str is not None:
      y_train_str = self._y_train_str[example_range]
      y_test_str = self._y_test_str[test_idxs]
    else:
      y_train_str = None
      y_test_str = None
    return Episode(x_train, y_train, train_indices, x_test, y_test, test_indices, x_unlabel, y_unlabel, unlabel_img_indices, y_train_str, y_test_str)

  @property
  def x_train(self):
    return self._x_train

  @property
  def train_indices(self):
    return self._train_indices

  @property
  def x_test(self):
    return self._x_test

  @property
  def test_indices(self):
    return self._test_indices

  @property
  def y_train(self):
    return self._y_train

  @property
  def y_test(self):
    return self._y_test

  @property
  def x_unlabel(self):
    return self._x_unlabel

  @property
  def y_unlabel(self):
    return self._y_unlabel

  @property
  def unlabel_indices(self):
    return self._unlabel_indices

  @property
  def y_train_str(self):
    return self._y_train_str

  @property
  def y_test_str(self):
    return self._y_test_str
