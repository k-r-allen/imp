# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
from fewshot.configs.config_factory import RegisterConfig


@RegisterConfig("cifar", "basic")
class BasicConfig(object):
  """Standard CNN on Omniglot with prototypical layer."""

  def __init__(self):
    self.name = "cifar_basic"
    self.model_class = "basic"
    self.height = 32
    self.width = 32
    self.num_channel = 3
    self.dim = 256
    self.update_clusters=False
    self.arch='basic'
    self.init_train_radius = 1.0
    self.init_radius = 1.0
    self.learn_train_radius = False
    self.learn_radius = False
    self.prior_weight = 1.#/self.dim
    mult = 1
    self.steps_per_valid = mult*2500
    self.ALPHA = 0.1
    self.steps_per_log = mult*2500
    self.steps_per_save = mult*2500
    self.learn_rate = 5e-4
    self.step_lr_every = 10000#mult*6000, 
    self.max_train_steps = mult*200000
    self.lr_decay_steps = range(20000, self.max_train_steps, self.step_lr_every)[1:]
    #self.lr_decay_steps = [mult*4000, mult*5000, mult*6000, mult*7000, mult*8000, mult*9000, mult*10000, mult*11000, mult*12000, mult*13000, mult*14000, mult*15000, 
    #mult*16000, mult*17000, mult*18000, mult*19000, mult*20000]
    
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(len(self.lr_decay_steps))))

@RegisterConfig("cifar", "protonet")
class BasicTestConfig(BasicConfig):

  def __init__(self):
    super(BasicTestConfig, self).__init__()

@RegisterConfig("cifar", "fully-supervised")
class FullySupervisedConfig(BasicConfig):

  def __init__(self):
    super(FullySupervisedConfig, self).__init__()
    self.name = 'fully-supervised'

@RegisterConfig("cifar", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.name = "cifar_kmeans-refine"
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1

    self.init_radius = 1.0
    self.init_train_radius = 1.0

@RegisterConfig("cifar", "kmeans-radius")
class KMeansConfig(BasicConfig):

  def __init__(self):
    super(KMeansConfig, self).__init__()
    self.model_class = "kmeans"
    self.num_cluster_steps = 1
    self.init_radius = 1.3
    self.init_train_radius = 1.0
    self.alpha = 0.5
    self.theta = 0.1
    self.learn_base = True
    self.name = "cifar-kmeans-learn-base-" + str(self.learn_base) + "-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius)

@RegisterConfig("cifar", "dp-means")
class DPMeansConfig(BasicConfig):

  def __init__(self):
    super(DPMeansConfig, self).__init__()
    self.model_class = "dp-means"
    self.num_cluster_steps = 1
    self.alpha = 0.01
    self.theta = 0.1
    self.init_lambda = 10.0
    self.lambda_decay = 0.99
    self.lambda_decay_steps = 1000
    self.init_radius = 1.0
    self.init_train_radius = 1.0
    self.name = "cifar-dp-means" + "-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius)


@RegisterConfig("cifar", "dp-means-multi-modal")
class DPMeansMultiModalConfig(BasicConfig):

  def __init__(self):
    super(DPMeansMultiModalConfig, self).__init__()

    self.model_class = "dp-means-multi-modal"
    self.num_cluster_steps = 1
    self.alpha = 0.01
    self.theta = 0.1
    self.use_lamda = False

    self.init_lambda = 20.0
    self.lambda_decay = 0.99    
    self.init_train_lambda = 10.0
    self.train_lambda_decay = 0.9

    self.lambda_decay_steps = 1000
    self.init_radius = 1.0
    self.ALPHA = 0.01
    self.learn_radius = False
    self.learn_train_radius = True
    self.name = "cifar-dp-means-multi-modal"+ "-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius)

@RegisterConfig("cifar", "dp-means-multi-modal-closest")
class DPMeansMultiModalClosestConfig(BasicConfig):

  def __init__(self):
    super(DPMeansMultiModalClosestConfig, self).__init__()

    self.model_class = "dp-means-multi-modal-closest"
    self.num_cluster_steps = 1
    self.alpha = 0.01
    self.theta = 0.1
    self.use_lamda = False

    self.init_lambda = 20.0
    self.lambda_decay = 0.99    
    self.init_train_lambda = 10.0
    self.train_lambda_decay = 0.9

    self.lambda_decay_steps = 1000
    self.init_radius = 1.0
    self.ALPHA = 0.01
    self.learn_radius = False
    self.learn_train_radius = True
    self.name = "cifar-dp-means-multi-modal-closest-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius)

@RegisterConfig("cifar", "dp-hierarchical")
class DPHierarchicalConfig(BasicConfig):

  def __init__(self):
    super(DPHierarchicalConfig, self).__init__()
    self.name = "cifar-dp-hierarchical"
    self.model_class = "dp-hierarchical"
    self.num_cluster_steps = 1
    self.alpha = 0.01
    self.theta = 0.1
    self.init_lambda = 20.0
    self.lambda_decay = 0.99
    self.lambda_decay_steps = 1000
    self.init_radius = 1.0

    self.init_global_lambda = 100.0
    self.lambda_decay = 0.98
    self.init_global_radius = 2.0
    self.learn_radius = False
    self.learn_global_radius = True

@RegisterConfig("cifar", "dp-cumulative")
class DPMeansCumulativeConfig(BasicConfig):

  def __init__(self):
    super(DPMeansCumulativeConfig, self).__init__()

    self.model_class = "dp-cumulative"
    self.num_cluster_steps = 1
    self.ALPHA = 0.1
    self.theta = 0.1
    self.init_lambda = 20.0
    self.lambda_decay = 0.7
    self.lambda_decay_steps = 3999
    self.init_radius = 1.0
    
    self.init_train_lambda = 20.0
    self.train_lambda_decay = 0.95
    self.dim = 64

    self.mem_iterations = 10
    self.memory_updates = 5000
    self.include_old_embeddings=True
    self.sample_support=True
    self.support_decay_steps=3999
    self.update_centroids=True
    self.sample_base=False
    self.learn_base = False
    self.set_size=0
    self.support_decay = 0.7
    self.support_discount = 0.1
    self.discount = 0.1
    self.support_multiplier = 1.05*self.learn_rate
    self.name = "cifar-dp-cumulative" + "-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius) + \
        "-mem-iterations-" + str(self.mem_iterations)

@RegisterConfig("cifar", "dp-multi-modal-cumulative")
class DPMeansMultiModalCumulativeConfig(BasicConfig):

  def __init__(self):
    super(DPMeansMultiModalCumulativeConfig, self).__init__()

    self.model_class = "dp-cumulative"
    self.num_cluster_steps = 1
    self.ALPHA = 0.5
    self.theta = 0.1
    self.init_lambda = 20.0
    self.lambda_decay = 0.7
    self.lambda_decay_steps = 3999
    self.init_radius = 1.0
    
    self.init_train_lambda = 20.0
    self.train_lambda_decay = 0.95
    self.dim = 64

    self.mem_iterations = 10
    self.memory_updates = 5000
    self.include_old_embeddings=True
    self.sample_support=True
    self.support_decay_steps=3999
    self.update_centroids=True
    self.off_target_weight = 0

    self.sample_base=False
    self.learn_base = False
    self.set_size=0
    self.support_decay = 0.7
    self.support_discount = 0.1
    self.discount = 0.1
    self.support_multiplier = 1.05*self.learn_rate
    self.name = "cifar-dp-cumulative" + "-learn-radius-" + str(self.learn_radius) + "-learn-train-radius-" + str(self.learn_train_radius) + \
        "-mem-iterations-" + str(self.mem_iterations)

