from fewshot.configs.config_factory import RegisterConfig


@RegisterConfig("mini-imagenet", "basic")
class BasicConfig(object):
  """Standard CNN with prototypical layer."""

  def __init__(self):
    self.name = "mini-imagenet_basic"
    self.model_class = "basic"
    self.num_channel = 3
    self.steps_per_valid = 2000
    self.steps_per_log = 100
    self.steps_per_save = 2000

    self.learn_rate = 1e-3
    self.dim = 1600
    self.ALPHA = 1e-5

    self.max_train_steps = 100000
    self.step_lr_every = 20000

    self.init_sigma_u = 1.0 
    self.learn_sigma_u = False
    self.init_sigma_l = 1.0
    self.learn_sigma_l = False

    self.lr_decay_steps = range(0, self.max_train_steps, self.step_lr_every)[1:]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(len(self.lr_decay_steps))))

@RegisterConfig("mini-imagenet", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1
    self.learn_sigma_l = False #True
    self.name = "mini-imagenet-kmeans-refine-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "kmeans-distractor")
class KMeansDistractorConfig(BasicConfig):

  def __init__(self):
    super(KMeansDistractorConfig, self).__init__()
    self.model_class = "kmeans-distractor"
    self.num_cluster_steps = 1

    self.init_sigma_u = 100
    self.init_sigma_l = 1.0
    self.learn_sigma_u = True
    self.learn_sigma_l = False

    self.name = "mini-imagenet-kmeans-distractor-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "imp")
class ImpModelConfig(BasicConfig):

  def __init__(self):
    super(ImpModelConfig, self).__init__()
    self.model_class = "imp"
    self.num_cluster_steps = 1

    self.init_sigma_u = 15.0 
    self.learn_sigma_u = True
    self.init_sigma_l = 15.0
    self.learn_sigma_l = True

    self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "crp")
class CRPConfig(BasicConfig):

  def __init__(self):
    super(CRPConfig, self).__init__()
    self.model_class = "crp"
    self.num_cluster_steps = 1
    self.eps = 1e-3
    self.ALPHA = 1e-5

    self.init_sigma_u = 15.0 
    self.learn_sigma_u = True
    self.init_sigma_l = 15.0
    self.learn_sigma_l = True

    self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "map-dp")
class MapDPConfig(BasicConfig):

  def __init__(self):
    super(MapDPConfig, self).__init__()
    self.model_class = "map-dp"
    self.num_cluster_steps = 1
    self.ALPHA = 1e-5

    self.init_sigma_u = 15.0 
    self.learn_sigma_u = True
    self.init_sigma_l = 15.0
    self.learn_sigma_l = True

    self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "soft-nn")
class SoftNNConfig(BasicConfig):

  def __init__(self):
    super(SoftNNConfig, self).__init__()
    self.model_class = "soft-nn"
    self.num_cluster_steps = 1

    self.init_sigma_u = 15.0 
    self.learn_sigma_u = True
    self.init_sigma_l = 15.0
    self.learn_sigma_l = True

    self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

@RegisterConfig("mini-imagenet", "dp-means-hard")
class DPMeansHardConfig(BasicConfig):

  def __init__(self):
    super(DPMeansHardConfig, self).__init__()
    self.model_class = "dp-means-hard"
    self.num_cluster_steps = 1

    self.init_sigma_u = 15.0 
    self.learn_sigma_u = False
    self.init_sigma_l = 15.0
    self.learn_sigma_l = False

    self.name = "mini-imagenet-" + self.model_class + "-learn-sigmau-" + str(self.learn_sigma_u) + "-learn-sigmal-" + str(self.learn_sigma_l)

