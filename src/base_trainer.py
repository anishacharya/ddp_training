import torch
from typing import Dict

from src.models import get_model
from src.dataloader import get_data_manager
from src.optimizers import get_optimizer, get_scheduler

from src.loss import get_loss


class TrainPipeline:
    """
    Base Trainer Class
    """
    def __init__(self, args, config: Dict, seed: int, repeat: int = 0):

        print('---- Fetching configs and Initializing stuff -----')
        self.ip_model_path = args.ip_model_path                     # path to checkpoint
        self.op_model_path = args.op_model_path                     # path to save model
        self.train_mode = args.train_mode                           # train mode: contrastive
        self.config = config
        self.repeat = repeat

        self.metrics = self.init_metric()
        self.metrics['config'] = config
        self.metrics['seed'] = seed

        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.curr_epoch = 0
        self.num_epochs = self.training_config.get('epochs')
        self.eval_freq = self.training_config.get('eval_freq')

        #  -----------------  Initialize Model -----------------
        self.model_arch = self.training_config.get("model_arch")
        self.proj_dim = self.training_config.get("proj_dim")
        self.num_classes = self.data_config.get('num_classes')

        self.model, self.metrics["num_param"] = get_model(nw_arch=self.model_arch,
                                                          proj_dim=self.proj_dim,
                                                          num_classes=self.num_classes)

        # ----------------- Load and process data -----------------
        self.data_manager = get_data_manager(data_config=self.data_config)

        # ----------------- initialize loss -----------------
        self.loss_fn = self.training_config.get('loss_fn')
        self.temp = self.training_config.get('temp')
        self.criterion = get_loss(loss_fn=self.loss_fn, temp=self.temp)

        # ----------------- initialize optimizer -----------------
        self.optimizer = get_optimizer(params=self.model.parameters(), optimizer_config=self.training_config)
        self.lrs = get_scheduler(optimizer=self.optimizer, lrs_config=self.training_config)
        self.lrs_freq = self.training_config.get('lrs_freq', 'epoch')

    @staticmethod
    def init_metric(verbose: bool = True):
        """
        :param verbose:
        :return:
        """
        metrics = \
            {
                "seed": 0,
                "num_param": 0,

                "run_best_val_acc": 0.0,

                "config": {},
                "train_loss": [],
                "val_acc": []
            }
        if verbose:
            print("Initialized Metrics : \n {}".format(metrics))
        return metrics

    def save_model(self):
        """
        save torch model
        """
        print('saving model at {}'.format(self.op_model_path))
        torch.save(self.model.state_dict(), self.op_model_path)

    def update_metric(self, train_loss: float, val_acc: float):
        """
        :param train_loss:
        :param val_acc:
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_acc'].append(val_acc)

        if val_acc > self.metrics['run_best_val_acc']:
            self.metrics['run_best_val_acc'] = val_acc
        print(
            "\n------------------------\n"
            "Classification Metrics:"
            "\n------------------------\n"
            "train loss      : {} \n"
            "val acc         : {}"
            "\n------------------------\n"
            "Best Val Acc    : {} \n"
            .format(train_loss,
                    val_acc,
                    self.metrics['run_best_val_acc']))
