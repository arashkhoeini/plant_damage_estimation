import torch
import time
import logging
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from data.prefetcher import DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
from models.unet import UNetVGG16, UNetVGG16MoCo, UNetVGG16PxCL
import utils
import os
import json
import math
import torch
import datetime
from typing import List, Dict, Any, Union, Optional, Tuple
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from sklearn import metrics as skmetrics


def get_instance(module, name, config, *args):
    """
    Get the corresponding class

    Args:
        module (str): the module to retrieve the class from
        name  (str): config item such that config[name]['type'] includes the class name
        config (Dictionary): configurations
        args (Dictionary): args to pass to the class

    Returns:
        an object of class config[name]['type']
    """
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


class Trainer:

    def __init__(
        self,
        pretrain_loss,
        train_loss,
        config,
        pretrain_loader,
        train_loader,
        test_loader=None,
        val_loader=None,
        train_logger=None,
        prefetch=True,
    ):

        self.config = config
        self.pretraining_method = config["pretraining"]["loss"]
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config["n_gpu"])

        self.num_classes = train_loader.dataset.num_classes
        self.model = self._get_model(availble_gpus)

        self.pretrain_loss = pretrain_loss

        self.train_loss = train_loss

        self.pretrain_loader = pretrain_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.start_epoch = 1

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose(
            [
                local_transforms.DeNormalize(
                    self.pretrain_loader.MEAN, self.pretrain_loader.STD
                ),
                transforms.ToPILImage(),
            ]
        )
        self.viz_transform = transforms.Compose(
            [transforms.Resize((400, 400)), transforms.ToTensor()]
        )

        self.pretrain_optimizer, self.finetune_optimizer = self._get_optimizers()
        self.pretrain_lr_scheduler = getattr(
            utils.lr_scheduler, config["pretraining"]["lr_scheduler"]["type"]
        )(
            self.pretrain_optimizer,
            config["pretraining"]["epochs"],
            len(pretrain_loader),
        )
        self.train_lr_scheduler = getattr(
            utils.lr_scheduler, config["finetuning"]["lr_scheduler"]["type"]
        )(self.finetune_optimizer, config["finetuning"]["epochs"], len(train_loader))

        if self.device == torch.device("cpu"):
            prefetch = False
        if prefetch:
            self.pretrain_loader = DataPrefetcher(
                self.pretrain_loader, device=self.device
            )
            self.train_loader = DataPrefetcher(self.train_loader, device=self.device)
            self.val_loader = DataPrefetcher(self.val_loader, device=self.device)
            self.test_loader = DataPrefetcher(self.test_loader, device=self.device)

        # CHECKPOINTS & TENSORBOARD
        self.output_dir, self.writer = self._get_output()

        # We need to initialize an Oracle object if the pre-training method is PxCL.
        # Oracle pseudo labels unlabeled data to select negative samples for pre-training
        if self.pretraining_method == "PxCL":
            self.oracle = UNetVGG16MoCo(self.num_classes, pretrained=False)
            if self.config["use_synch_bn"]:
                self.oracle = convert_model(self.oracle)
                self.oracle = DataParallelWithCallback(
                    self.oracle, device_ids=availble_gpus
                )
            else:
                self.oracle = torch.nn.DataParallel(
                    self.oracle, device_ids=availble_gpus
                )
            # self.oracle.load_state_dict(torch.load(os.path.join(self.config['save_dir'], 'oracle.pth'), map_location=torch.device('cpu'))['state_dict'])
            self.oracle.load_state_dict(
                torch.load(
                    "oracle.pth",
                    weights_only=False,
                )["state_dict"]
            )
            self.oracle.to(self.device)

        # if resume: self._resume_checkpoint(resume)

        torch.backends.cudnn.benchmark = True

    def pretrain(self):
        """
        Pre-trains the model using contrastive loss and unlabaled data
        """
        for epoch in range(self.start_epoch, self.config["pretraining"]["epochs"] + 1):
            # RUN TRAIN (AND VAL)
            results = self._pretrain_epoch(epoch)

            # SAVE CHECKPOINT
            if (epoch % self.config["pretraining"]["save_period"] == 0) or (
                epoch == self.config["pretraining"]["epochs"]
            ):
                # if epoch == self.config['pretraining']['epochs']:
                #     self.config["state"] = 'train'
                filename = self._save_checkpoint(epoch)

        return filename

    def train(self):
        """
        Fine-tunes the pre-trained model using labeled data
        """
        for epoch in range(self.start_epoch, self.config["finetuning"]["epochs"] + 1):
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch)
            if epoch % self.config["finetuning"]["val_per_epochs"] == 0:
                results = self._valid_epoch(testing=False, epoch=epoch)
                if self.train_logger is not None:
                    log = {"epoch": epoch, **results}
                    self.train_logger.add_entry(log)

                self.logger.info(f"\n         ## Info for epoch {epoch} ## ")
                for k, v in results.items():
                    self.logger.info(f"         {str(k):15s}: {v}")

        # Save checkpoint only if we are training using the whole labeled dataset
        if not self.config["testing"]["cross_validation"]:
            self._save_checkpoint(epoch)

    def test(self):
        """
        Tests the fine-tuned model
        """
        return self._valid_epoch(testing=True)

    def _save_checkpoint(self, epoch):
        """
        Saves the model checkpoint

        Args:
            epoch (int): current epoch

        Returns:
            path of the saved checkpoint
        """

        checkpoint = {"config": self.config, "state_dict": self.model.state_dict()}

        filename = os.path.join(self.output_dir, f"checkpoint_epoch{epoch}.pth")
        self.logger.info(f"\nSaving a checkpoint: {filename} ...")
        torch.save(checkpoint, filename)

        return filename

    def _resume_checkpoint(self, resume_path):
        """
        Resumes a checkpoint stored in resume_path

        Args:
            resume_path (str): path of stored checkpoint
        """
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        if "state_dict" in checkpoint.keys():
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.logger.info(
            f"Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded"
        )

    def _train_epoch(self, epoch):
        """
        Contains logic for training (fine-tuning) the model for one epoch

        Args:
            epoch (int): epoch number

        Returns:
            log
        """
        self.logger.info("\n")

        self.model.train()

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            target[target == 255] = 1
            target[target == 120] = 2
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            self.finetune_optimizer.zero_grad()
            output = self.model(data)
            assert output.size()[2:] == target.size()[1:]  # type: ignore
            assert output.size()[1] == self.num_classes
            # print("##"*10)
            # from collections import Counter
            # # print(Counter(output.cpu().detach().numpy().reshape(-1)))
            # print(Counter(target.cpu().detach().numpy().reshape(-1)))
            loss = self.train_loss(output, target)

            if isinstance(self.train_loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.finetune_optimizer.step()
            self.train_lr_scheduler.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            final_metrics = self._get_seg_metrics()

            # PRINT INFO
            tbar.set_description(
                f"TRAIN ({epoch}) | Loss: {self.total_loss.average:.3f} | Acc {final_metrics['pixel_accuracy']:.2f} mIoU {final_metrics['mean_IoU']:.2f} | B {self.batch_time.average:.2f} D {self.data_time.average:.2f} |"
            )

        # LOGGING & TENSORBOARD
        # if epoch % self.config['log_per_iter'] == 0:
        # self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
        self.writer.add_scalar(f"loss", loss.item(), epoch)
        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        # for k, v in list(seg_metrics.items())[:-1]:
        #     self.writer.add_scalar(f'train/{k}', v, epoch)
        for i, opt_group in enumerate(self.finetune_optimizer.param_groups):
            self.writer.add_scalar(f"Learning_rate_{i}", opt_group["lr"], epoch)

        # RETURN LOSS & METRICS
        log = {"loss": self.total_loss.average, **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _pretrain_epoch(self, epoch):
        """
        Contains logic for pre-training the model for one epoch

        Args:
            epoch (int): epoch number

        Returns:
            log
        """
        self.logger.info("\n")

        self.model.train()

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.pretrain_loader, ncols=130)
        for batch_idx, (view1, view2) in enumerate(tbar):
            assert view1.size() == view2.size()  # type: ignore

            self.data_time.update(time.time() - tic)
            view1, view2 = view1.to(self.device), view2.to(self.device)  # type: ignore

            # LOSS & OPTIMIZE
            self.pretrain_optimizer.zero_grad()
            if self.pretraining_method == "SimSiam":
                z1, p1 = self.model(view1, pretraining=True)
                z2, p2 = self.model(view2, pretraining=True)
                loss = self.pretrain_loss(z1, p1, z2, p2)
            elif self.pretraining_method == "MoCo":
                logits, labels = self.model(view1, view2, pretraining=True)
                loss = self.pretrain_loss(logits, labels)
            elif self.pretraining_method == "PxCL":
                z1 = self.model(view1, pretraining=True)
                z2 = self.model(view2, pretraining=True)
                labels = self.oracle(view1)
                loss = self.pretrain_loss(z1, z2, labels)

            if isinstance(self.pretrain_loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.pretrain_optimizer.step()
            self.pretrain_lr_scheduler.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()
            # PRINT INFO
            tbar.set_description(
                "TRAIN ({}) | Loss: {:.3f} |".format(epoch, self.total_loss.average)
            )

        # TENSORBOARD
        self.writer.add_scalar(f"pretrain/loss", loss.item(), epoch)
        for i, opt_group in enumerate(self.pretrain_optimizer.param_groups):
            self.writer.add_scalar(
                f"pretrain/Learning_rate_{i}", opt_group["lr"], epoch
            )
        # RETURN LOSS & METRICS
        log = {"loss": self.total_loss.average}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, testing, epoch=None):
        """
        Contains logic for validation (testing) the model for one epoch

        Args:
            testing (bool): indicates whether this is for validation or for testing.
            epoch (int): epoch number

        Returns:
            Performance stats of the model on the validation or test set.
        """
        assert testing == (epoch is None)
        if testing:
            epoch = 0
        if self.val_loader is None:
            self.logger.warning(
                "No data loader was passed for the validation step, No validation is performed !"
            )
            return {}
        if not testing:
            self.logger.info("\n###### EVALUATION ######")
        else:
            self.logger.info("\n###### TESTING ######")

        self.model.eval()

        self._reset_metrics()
        if testing:
            tbar = tqdm(self.test_loader, ncols=130)
        else:
            tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            target_list = []
            output_list = []
            for batch_idx, (data, target) in enumerate(tbar):
                target[target == 255] = 1
                target[target == 120] = 2
                # data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)

                loss = self.train_loss(output, target)
                if isinstance(self.train_loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)
                target_list.append(target)
                output_list.append(output)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()  # type: ignore
                    output_np = output.data.max(1)[1].cpu().numpy()
                    output_np[output_np == 1] = 255
                    output_np[output_np == 2] = 120
                    target_np[target_np == 1] = 255
                    target_np[target_np == 2] = 120
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                final_metrics = self._get_seg_metrics()
                tbar.set_description(
                    f"EVAL ({epoch}) | Loss: {self.total_loss.average:.3f}, PixelAcc: {final_metrics['pixel_accuracy']:.2f}, Mean IoU: {final_metrics['mean_IoU']:.2f} |"
                )

            if not testing:
                wrt_mode = "val"
            else:
                wrt_mode = "test"

            if testing or (epoch % self.config["finetuning"]["save_period"] == 0):
                # WRTING & VISUALIZING THE MASKS
                val_img = []
                palette = self.train_loader.dataset.palette
                for i, (d, t, o) in enumerate(val_visual):
                    d = self.restore_transform(d)
                    t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                    d, t, o = d.convert("RGB"), t.convert("RGB"), o.convert("RGB")
                    [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                    val_img.extend([d, t, o])
                val_img = torch.stack(val_img, 0)
                val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
                self.writer.add_image(
                    f"{wrt_mode}/inputs_targets_predictions", val_img, epoch
                )

            seg_metrics = self._get_seg_metrics()
            # METRICS TO TENSORBOARD
            # if not testing:
            # wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f"{wrt_mode}/loss", self.total_loss.average, epoch)

            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f"{wrt_mode}/{k}", v, epoch)

            scores = self._compute_scores(output_list, target_list)

            log = {"val_loss": self.total_loss.average, **seg_metrics, **scores}

        return log

    def _reset_metrics(self):
        """
        resets metrics
        """
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        """
        Updates segmentation metrics
        """
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        """
        Returns:
            segmentation metrics
        """
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        result = {
            "pixel_accuracy": np.round(pixAcc, 3),
            "mean_IoU": np.round(mIoU, 3),
        }
        for i, iou in enumerate(IoU):
            result[f"class{i}_IoU"] = np.round(iou, 3)

        return result

    def _get_available_devices(self, n_gpu):
        """
        Returns:
            the current device
            list of available devices
        """
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning("No GPUs detected, using the CPU")
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(
                f"Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available"
            )
            n_gpu = sys_gpu

        device = torch.device("cuda:0" if n_gpu > 0 else "cpu")
        self.logger.info(f"Detected GPUs: {sys_gpu} Requested: {n_gpu}")
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def _get_model(self, availble_gpus, pretrained=False):
        """
        Creates an object of the model based on the value stored in self.pretraining_method

        Args:
            available_gpus (List): list of available GPUs. Use self._get_available_devices() to retrieve this list
            pretrained (bool): whether we want a pretrained model or not. Note that pretraining here indicates to pretrained using supervised learining on ImageNet dataset

        Returns
            torch.nn.Module: model object
        """
        if self.pretraining_method == "SimSiam":
            model = UNetVGG16(self.num_classes, pretrained=pretrained)
        elif self.pretraining_method == "MoCo":
            model = UNetVGG16MoCo(self.num_classes, pretrained=pretrained)
        elif self.pretraining_method == "PxCL":
            model = UNetVGG16PxCL(self.num_classes, pretrained=pretrained)
        if self.config["use_synch_bn"]:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=availble_gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=availble_gpus)
        model.to(self.device)
        return model

    def _get_optimizers(self):
        """
        Returns:
            pretraing and traing optimizers
        """
        if isinstance(self.model, torch.nn.DataParallel):
            if self.pretraining_method == "PxCL":
                pretrain_params = filter(
                    lambda p: p.requires_grad, self.model.module.parameters()
                )
            else:
                pretrain_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.get_encoder_params(),
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.projection_head.parameters(),
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.prediction_head.parameters(),  # type: ignore
                        )
                    },
                ]
            if self.config["finetune_optimizer"]["differential_lr"]:
                train_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.get_decoder_params(),
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.get_encoder_params(),
                        ),
                        "lr": self.config["finetune_optimizer"]["args"]["lr"] / 10,
                    },
                ]
            else:
                train_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.get_decoder_params(),
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.module.get_encoder_params(),
                        )
                    },
                ]

        else:
            if self.pretraining_method == "PxCL":
                pretrain_params = filter(
                    lambda p: p.requires_grad, self.model.module.parameters()
                )
            else:
                pretrain_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.model.get_encoder_params()
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.projection_head.parameters(),
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.prediction_head.parameters(),
                        )
                    },
                ]

            if self.config["finetune_optimizer"]["differential_lr"]:
                train_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.model.get_decoder_params()
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.model.get_encoder_params()
                        ),
                        "lr": self.config["finetune_optimizer"]["args"]["lr"] / 10,
                    },
                ]
            else:
                train_params = [
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.model.get_decoder_params()
                        )
                    },
                    {
                        "params": filter(
                            lambda p: p.requires_grad, self.model.get_encoder_params()
                        )
                    },
                ]

        pretrain_optimizer = get_instance(
            torch.optim, "pretrain_optimizer", self.config, pretrain_params
        )
        finetune_optimizer = get_instance(
            torch.optim, "finetune_optimizer", self.config, train_params
        )
        return pretrain_optimizer, finetune_optimizer

    def _get_output(self):
        """
        Returns:
            str: output directory
            tensorboard.SummaryWriter: writer object to write on tensorboard
        """
        start_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

        if "output_dir" in self.config:
            output_dir = self.config["output_dir"]
            writer_dir = self.config["writer_dir"]
        else:
            output_dir = os.path.join(
                self.config["save_dir"], self.config["name"], start_time
            )
            writer_dir = os.path.join(
                self.config["log_dir"], self.config["name"], start_time
            )
            self.config["output_dir"] = output_dir
            self.config["writer_dir"] = writer_dir
            helpers.dir_exists(output_dir)

        config_save_path = os.path.join(output_dir, "config.json")
        with open(config_save_path, "w") as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer = tensorboard.SummaryWriter(writer_dir)

        return output_dir, writer

    def _compute_scores(self, output_list, target_list):
        """
        Computes precision, recall and f1-score

        Args:
            output_list (torch.Tensor): predictions by the model
            target_list (torch.Tensor): ground truth targets

        Returns:
            Dictionary: computed scores

        """
        damage_rmse = 0
        for i, output in enumerate(output_list):
            n_leaf_pixels = torch.sum(output[output == 1])
            n_damage_pixels = torch.sum(output[output == 2])
            print(n_damage_pixels, n_leaf_pixels)
            damage_ratio_pred = n_damage_pixels / (n_damage_pixels + n_leaf_pixels)

            target = target_list[i]
            n_leaf_pixels = torch.sum(target[target == 1])
            n_damage_pixels = torch.sum(target[target == 2])
            print(n_damage_pixels, n_leaf_pixels)
            damage_ratio_gt = n_damage_pixels / (n_damage_pixels + n_leaf_pixels)

            damage_rmse += (damage_ratio_gt - damage_ratio_pred) ** 2

        damage_rmse = torch.sqrt(damage_rmse / len(output_list))  # type: ignore

        output = torch.cat(output_list, dim=0)
        target = torch.cat(target_list, dim=0)

        _, predict = torch.max(output.data, 1)

        target = target.detach().cpu().numpy().reshape(-1)
        predict = predict.detach().cpu().numpy().reshape(-1)

        precision = skmetrics.precision_score(target, predict, average=None)
        recall = skmetrics.recall_score(target, predict, average=None)
        f1score = skmetrics.f1_score(target, predict, average=None)

        scores = {"damage_rmse": damage_rmse}

        scores["precision"] = sum(precision) / len(precision)  # type: ignore
        scores["recall"] = sum(recall) / len(recall)  # type: ignore
        scores["f1score"] = sum(f1score) / len(f1score)  # type: ignore

        for i in range(len(precision)):  # type: ignore
            scores[f"precision_{i}"] = precision[i]  # type: ignore
            scores[f"recall_{i}"] = recall[i]  # type: ignore
            scores[f"f1score_{i}"] = f1score[i]  # type: ignore

        return scores
