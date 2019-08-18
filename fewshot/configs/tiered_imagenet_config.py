from fewshot.configs.config_factory import RegisterConfig


@RegisterConfig("tiered-imagenet", "basic")
class BasicConfig(object):
  """Standard CNN on tiered-imagenet with prototypical layer."""

  def __init__(self):
    self.name = "tiered-imagenet_basic"
    self.model_class = "basic"

    self.num_channel = 3
    self.dim = 1600
    self.ALPHA = 0.1

    self.init_sigma_l = 1.0
    self.init_sigma_u = 1.0

    self.learn_sigma_l = False
    self.learn_sigma_u = False

    self.steps_per_valid = 2000
    self.steps_per_log = 1000
    self.steps_per_save = 2000

    self.learn_rate = 1e-3
    self.lr_scheduler = "fixed"
    self.max_train_steps = 200000
    self.lr_decay_steps = list(range(0, self.max_train_steps, 25000)[1:])
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x, range(
            len(self.lr_decay_steps))))

@RegisterConfig("tiered-imagenet", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1
    self.init_sigma_l = 1.0
    self.learn_sigma_l = False #True
    self.name = "tiered-imagenet-kmeans-refine-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "kmeans-distractor")
class KMeansDistractorConfig(BasicConfig):

  def __init__(self):
    super(KMeansDistractorConfig, self).__init__()
    self.model_class = "kmeans-distractor"
    self.num_cluster_steps = 1

    self.init_sigma_u = 100
    self.init_sigma_l = 1.0
    self.learn_sigma_u = True
    self.learn_sigma_l = False

    self.name = "tiered-imagenet-kmeans-distractor-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "imp")
class ImpModelConfig(BasicConfig):

  def __init__(self):
    super(ImpModelConfig, self).__init__()
    self.model_class = "imp"
    self.num_cluster_steps = 1

    self.learn_sigma_u = False
    self.learn_sigma_l = True

    self.name = "tiered-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "crp")
class CRPConfig(BasicConfig):

  def __init__(self):
    super(CRPConfig, self).__init__()
    self.model_class = "crp"
    self.num_cluster_steps = 1
    self.eps = 1e-3
    self.ALPHA = 1.

    self.name = "tiered-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "map-dp")
class MapDPConfig(BasicConfig):

  def __init__(self):
    super(MapDPConfig, self).__init__()
    self.model_class = "map-dp"
    self.num_cluster_steps = 1
    self.ALPHA = 1.

    self.name = "tiered-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "soft-nn")
class SoftNNConfig(BasicConfig):

  def __init__(self):
    super(SoftNNConfig, self).__init__()
    self.model_class = "soft-nn"
    self.num_cluster_steps = 1

    self.name = "tiered-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("tiered-imagenet", "dp-means-hard")
class DPMeansHardConfig(BasicConfig):

  def __init__(self):
    super(DPMeansHardConfig, self).__init__()
    self.model_class = "dp-means-hard"
    self.num_cluster_steps = 1

    self.learn_sigma_u = False
    self.learn_sigma_l = False

    self.name = "tiered-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

