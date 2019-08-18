import os


DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
  """Registers a dataset class"""

  def decorator(f):
    DATASET_REGISTRY[dataset_name] = f
    return f

  return decorator


def get_data_folder(data_root, dataset_name):
  data_folder = os.path.join(data_root, dataset_name)
  return data_folder


def get_dataset(FLAGS, dataset_name, split, *args, **kwargs):
  if dataset_name in DATASET_REGISTRY:
    return DATASET_REGISTRY[dataset_name](FLAGS, get_data_folder(FLAGS.data_root, dataset_name), split,
                                          *args, **kwargs)
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))
