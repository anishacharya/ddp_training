from tqdm import tqdm
import torch
from src.eval import basic_eval
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn

from src.optimizers import get_optimizer, get_scheduler
from src.base_trainer import TrainPipeline
from numpyencoder import NumpyEncoder
import json


def finetune(args, config):
    """

    :param args:
    :param config:
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        world_size = n_gpus
        mp.spawn(finetune_ddp,
                 args=(world_size, args, config, ),
                 nprocs=world_size,
                 join=True)

    else:
        # Training - Repeat over the random seeds
        results = []
        curr_best_val_acc = 0  # This captures best eval across seeds * across epochs

        for ix in np.arange(args.n_repeat):
            # ensure reproducibility
            seed = int((ix + 1) * 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            cudnn.deterministic = True
            torch.autograd.set_detect_anomaly(True)

            trainer = TrainPipeline(args=args, config=config, seed=seed, repeat=ix)
            data_manager = trainer.data_manager

            tr_dataset, te_dataset = data_manager.get_dataset()
            tr_dataset.transform = data_manager.sv_transform
            te_dataset.transform = data_manager.basic_transform

            finetune_single_device(trainer=trainer,
                                   data_manager=data_manager,
                                   tr_dataset=tr_dataset,
                                   te_dataset=te_dataset,
                                   curr_best_val_acc=curr_best_val_acc)

            # End of training for one seed
            if trainer.metrics['run_best_val_acc'] > curr_best_val_acc:
                curr_best_val_acc = trainer.metrics['run_best_val_acc']
                print('Best Accuracy across seeds {}'.format(curr_best_val_acc))

            results.append(trainer.metrics)
            del trainer  # free up trainer - possibly unnecessary

        # Log results
        best_acc = np.array([run['run_best_val_acc'] for run in results])
        std = np.std(best_acc, axis=0)
        mean_best_acc = np.mean(best_acc)
        print('Best run validation accuracy: {}'.format(curr_best_val_acc))
        print('Validation Accuracy: {} +- {}'.format(mean_best_acc * 100, std * 100))

        # Write metrics
        with open(args.log_path, 'w+') as f:
            print('saving log file')
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def finetune_ddp(rank, world_size, args, config):
    ddp_setup(rank, world_size, args.port)
    device = torch.device("cuda", rank)

    # Training - Repeat over the random seeds
    results = []
    curr_best_val_acc = 0  # This captures best eval across seeds * across epochs

    for ix in np.arange(args.n_repeat):
        # ensure reproducibility
        seed = int((ix + 1) * 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)

        trainer = TrainPipeline(args=args, config=config, seed=seed, repeat=ix)
        data_manager = trainer.data_manager

        tr_dataset, te_dataset = data_manager.get_dataset()
        tr_dataset.transform = data_manager.sv_transform
        te_dataset.transform = data_manager.basic_transform

        ddp_tr_sampler = DistributedSampler(tr_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True,
                                            seed=seed)

        ddp_te_sampler = DistributedSampler(te_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=True,
                                            seed=seed)

        tr_loader = DataLoader(dataset=tr_dataset,
                               batch_size=data_manager.train_batch_size,
                               sampler=ddp_tr_sampler,
                               num_workers=data_manager.num_worker)

        te_loader = DataLoader(dataset=te_dataset,
                               batch_size=data_manager.test_batch_size,
                               sampler=ddp_te_sampler,
                               num_workers=data_manager.num_worker)
        trainer.model.to(device)
        trainer.model = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
        trainer.model = DistributedDataParallel(trainer.model, device_ids=[rank], find_unused_parameters=True)

        trainer.optimizer = get_optimizer(params=trainer.model.parameters(), optimizer_config=trainer.training_config)
        trainer.lrs = get_scheduler(optimizer=trainer.optimizer, lrs_config=trainer.training_config)
        trainer.lrs_freq = trainer.training_config.get('lrs_freq', 'epoch')

        while trainer.curr_epoch < trainer.num_epochs:
            p_bar = tqdm(total=len(tr_loader))
            p_bar.set_description('(Repeat = {} Rank = {} ::Epoch = {}/{}, lr: {}), '.format(
                trainer.repeat,
                rank,
                trainer.curr_epoch + 1,
                trainer.num_epochs,
                trainer.optimizer.param_groups[0]['lr']
            ))
            train_loss, curr_iter = 0, 0

            trainer.model.train()
            ddp_tr_sampler.set_epoch(trainer.curr_epoch)

            for batch_ix, (feature, labels) in enumerate(tr_loader):
                feature = feature.to(device)
                labels = labels.to(device)
                trainer.optimizer.zero_grad()
                z = trainer.model(feature=feature, mode='finetune')
                loss = trainer.criterion(z, labels)
                loss.backward()
                trainer.optimizer.step()
                p_bar.set_postfix({'train loss': loss.item()}), p_bar.update()
                train_loss += loss.item()
                curr_iter += 1
            train_loss /= curr_iter
            trainer.lrs and trainer.lrs.step()

            if (trainer.curr_epoch + 1) % trainer.eval_freq == 0:
                dist.barrier()
                trainer.model.eval()

                correct, total = 0, 0
                with torch.no_grad():
                    for data, target in te_loader:
                        data = data.to(device)
                        target = target.to(device)
                        output = trainer.model(data, mode='finetune')
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)

                val_acc = torch.tensor(correct / total)
                dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
                val_acc /= world_size

                # print(f"Epoch {trainer.curr_epoch}, Train Loss: {train_loss}, Test Accuracy: {val_acc.item()}")
                trainer.update_metric(train_loss=train_loss, val_acc=val_acc.item())

                # print(trainer.metrics)
                if val_acc > curr_best_val_acc:
                    curr_best_val_acc = val_acc  # This is the best across all seeds
                    if rank == 0:
                        print('saving model at {}'.format(trainer.op_model_path))
                        torch.save(trainer.model.module.state_dict(), trainer.op_model_path)
                        # trainer.save_model()

            # Proceed to Next epoch
            trainer.curr_epoch += 1

        # End of training for one seed
        if trainer.metrics['run_best_val_acc'] > curr_best_val_acc:
            curr_best_val_acc = trainer.metrics['run_best_val_acc']
            print('Best Accuracy across seeds {}'.format(curr_best_val_acc))

        results.append(trainer.metrics)
        del trainer  # free up trainer - possibly unnecessary

    # End of training for all seeds
    # Log results
    if rank == 0:
        best_acc = np.array([run['run_best_val_acc'] for run in results])
        std = np.std(best_acc, axis=0)
        mean_best_acc = np.mean(best_acc)
        print('Best run validation accuracy: {}'.format(curr_best_val_acc))
        print('Validation Accuracy: {} +- {}'.format(mean_best_acc * 100, std * 100))

        # Write metrics
        with open(args.log_path, 'w+') as f:
            print('saving log file')
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    ddp_cleanup()


def finetune_single_device(trainer, data_manager, tr_dataset, te_dataset, curr_best_val_acc):
    # wrap around DataLoader
    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=data_manager.train_batch_size,
                           pin_memory=True,
                           shuffle=True,
                           num_workers=data_manager.num_worker)

    te_loader = DataLoader(dataset=te_dataset,
                           batch_size=data_manager.test_batch_size,
                           pin_memory=True,
                           shuffle=True,
                           num_workers=data_manager.num_worker)

    # ------------- Training Loop ------------- #
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if trainer.ip_model_path:
        print('Loading Pretrained Model : {}'.format(trainer.ip_model_path))
        trainer.model.load_state_dict(torch.load(trainer.ip_model_path, map_location=device))

    while trainer.curr_epoch < trainer.num_epochs:
        p_bar = tqdm(total=len(tr_loader))
        p_bar.set_description('(Epoch = {} : {}/{}, lr: {}), Training Progress :'
                              .format(trainer.repeat,
                                      trainer.curr_epoch + 1,
                                      trainer.num_epochs,
                                      trainer.optimizer.param_groups[0]['lr']))
        train_loss, curr_iter = 0, 0
        if torch.cuda.is_available():
            trainer.model.to(device)  # cuda()
            trainer.criterion.to(device)  # cuda()

        #  ----------------  Run one epoch of training  ----------------- #
        for batch_ix, (feature, labels) in enumerate(tr_loader):
            feature = feature.to(device)  # cuda()
            labels = labels.to(device)  # cuda()
            trainer.optimizer.zero_grad()
            z = trainer.model(feature=feature, mode='finetune')
            loss = trainer.criterion(z, labels)
            # if torch.cuda.is_available():
            loss = loss.to(device)  # cuda()
            loss.backward()
            trainer.optimizer.step()
            p_bar.set_postfix({'train loss': loss.item()}), p_bar.update()
            train_loss += loss.item()
            curr_iter += 1
        train_loss /= curr_iter
        trainer.lrs and trainer.lrs.step()

        # Evaluate on Test data
        if (trainer.curr_epoch + 1) % trainer.eval_freq == 0:
            val_acc = basic_eval(model=trainer.model, data_loader=te_loader, mode='finetune')
            trainer.update_metric(train_loss=train_loss, val_acc=val_acc)

            if val_acc > curr_best_val_acc:
                trainer.save_model()
                curr_best_val_acc = val_acc  # This is the best across all seeds

        # Proceed to Next epoch
        trainer.curr_epoch += 1


# DDP Related -------
def ddp_setup(rank, world_size, port:str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()
