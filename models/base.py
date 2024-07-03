import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call
                                            BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply
                                            preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and
                                            update network weights.
    """

    def __init__(self, config):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags;
                                 needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own
        initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses
                                                    that you want to plot and
                                                    save.
            -- self.model_names (str list):         specify the images that you
                                                    want to display and save.
            -- self.visual_names (str list):        define networks used in our
                                                    training.
            -- self.optimizers (optimizer list):    define and initialize
                                                    optimizers. You can define
                                                    one optimizer for each
                                                    network. If two networks
                                                    are updated at the same
                                                    time, you can use
                                                    itertools.chain to group
                                                    them.
                                                    See cycle_gan_model.py for
                                                    an example.
        """
        self.config = config
        self.device = config["DEVICE"]
        self.path_train = os.path.join(
            config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "train"
        )
        self.path_test = os.path.join(
            config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"], "test"
        )

        for path in [self.path_train, self.path_test]:
            os.makedirs(path, exist_ok=True)

        self.loss_names = []
        self.model_names = []

        pass

    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata
            information.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass; called by both functions
        <optimize_parameters> and <test>.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights;
        called in every training iteration
        """
        pass

    @abstractmethod
    def inference(self):
        pass

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def compute_visuals(self):
        """
        Calculate additional output images for
        visdom and HTML visualization
        """
        pass

    def update_learning_rate(self):
        """
        Update learning rates for all the networks;
        called at the end of every epoch
        """
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """
        Return traning losses / errors. train.py will print out
        these errors on console, and save them to a file
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name
            '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{epoch}_{name}.pth"
                save_path = os.path.join(self.path_train, save_filename)
                net = getattr(self, name)

                torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch):
        """
        Load all the networks from the disk.
        Allow replacement of different instance normalization modules

        Parameters:
            epoch (int) -- current epoch; used in the file name
            '%s_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_{name}.pth"
                load_path = os.path.join(self.path_train, load_filename)
                net = getattr(self, name)

                checkpoint = torch.load(load_path, map_location=self.device)
                model_dict = net.state_dict()
                checkpoint = {
                    k: v for k, v in checkpoint.items() if k in model_dict
                }
                model_dict.update(checkpoint)
                net.load_state_dict(model_dict)

    def print_networks(self, verbose):
        """
        Print the total number of parameters in the network
        and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    f"[Network {name}] Total number of "
                    f"parameters : {(num_params / 1e6):.3f} M"
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid
        unnecessary computations

        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require
                                     gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def data_dependent_initialize(self, data):
        pass

    def setup(self):
        pass
