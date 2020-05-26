from create_data_files import PATH_RSICD
import torch
from torchvision import transforms, models
from torch import nn
from datasets import ClassificationDataset
from torch.utils.data import DataLoader
from utils.early_stop import EarlyStopping
from optimizer import get_optimizer
import logging
import os
import numpy as np
import time
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

DISABLE_STEPS = False
#FILE_NAME = "classification_efficientnet_focalloss"
FILE_NAME = "classification_resnetfreq"
FINE_TUNE = True
EFFICIENT_NET = False
FREQ = False
FOCAL_LOSS = True
EPOCHS = 300
BATCH_SIZE = 8
EPOCHS_LIMIT_WITHOUT_IMPROVEMENT = 5


NUM_WORKERS = 0
OPTIMIZER_TYPE = "adam"
OPTIMIZER_LR = 1e-4


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)

    # model = nn.Sequential(*list(model.children())[:])

    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Change the stride from 2 to 1 to match the input size
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetDCT_Upscaled_Static(nn.Module):
    def __init__(self, channels=0, pretrained=True, input_gate=False):
        super(ResNetDCT_Upscaled_Static, self).__init__()

        self.input_gate = input_gate

        model = resnet50(pretrained=pretrained)

        self.model = nn.Sequential(*list(model.children())[4:-1])
        self.fc = list(model.children())[-1]
        self.relu = nn.ReLU(inplace=True)

        if channels == 0 or channels == 192:
            out_ch = self.model[0][0].conv1.out_channels
            self.model[0][0].conv1 = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].conv1)

            out_ch = self.model[0][0].downsample[0].out_channels
            self.model[0][0].downsample[0] = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].downsample[0])

            # temp_layer = conv3x3(channels, out_ch, 1)
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].conv1.weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].conv1 = temp_layer

            # out_ch = self.model[0][0].downsample[0].out_channels
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].downsample[0].weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].downsample[0] = temp_layer
        elif channels < 64:
            out_ch = self.model[0][0].conv1.out_channels
            # temp_layer = conv3x3(channels, out_ch, 1)
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].conv1.weight.data[:, :channels]
            self.model[0][0].conv1 = temp_layer

            out_ch = self.model[0][0].downsample[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].downsample[0].weight.data[:, :channels]
            self.model[0][0].downsample[0] = temp_layer

        if input_gate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_gate:
            x, inp_atten = self.inp_GM(x)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)  # 2048
        x = self.fc(x)
        if self.input_gate:
            return x, inp_atten
        else:
            return x


class ClassificationModel():
    MODEL_DIRECTORY = "experiments/results/"

    def __init__(self, vocab_size, device):
        self.device = device
        self.checkpoint_exists = False

        if EFFICIENT_NET:
            image_model = EfficientNet.from_pretrained('efficientnet-b4')
            num_features = image_model._fc.in_features
            image_model._fc = nn.Linear(num_features, vocab_size)
        elif FREQ:
            channels = 192
            image_model = ResNetDCT_Upscaled_Static(channels=channels)
            print("image_model", image_model)
            num_features = image_model.fc.in_features
            image_model.fc = nn.Linear(num_features, vocab_size)
            checkpoint = torch.load('src/resnet50dct_upscaled_static_64/model_best.pth.tar.pth.tar')
            image_model.load_state_dict(checkpoint['model'])

        else:  # use densenet
            image_model = models.densenet201(pretrained=True)
            num_features = image_model.classifier.in_features
            image_model.classifier = nn.Linear(num_features, vocab_size)

        self.model = image_model.to(self.device)

    def setup_to_train(self):
        if FOCAL_LOSS:
            self.criterion = FocalLoss().to(self.device)
        else:
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        if FINE_TUNE:
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.parameters(), OPTIMIZER_LR)
        else:  # ConvNet as fixed feature extractor (freeze all the network except the final layer)
            self.optimizer = get_optimizer(OPTIMIZER_TYPE, self.model.classifier.parameters(), OPTIMIZER_LR)

        self._load_weights_from_checkpoint(load_to_train=True)

    def train_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)

        loss = self.criterion(outputs, targets)

        self.model.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss

    def val_step(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)
        loss = self.criterion(outputs, targets)

        return loss

    def train(self, train_dataloader, val_dataloader, print_freq=5):
        early_stopping = EarlyStopping(
            epochs_limit_without_improvement=EPOCHS_LIMIT_WITHOUT_IMPROVEMENT,
            epochs_since_last_improvement=self.checkpoint_epochs_since_last_improvement
            if self.checkpoint_exists else 0,
            baseline=self.checkpoint_val_loss if self.checkpoint_exists else np.Inf,
            encoder_optimizer=None,  # TENS
            decoder_optimizer=None,
            period_decay_lr=1000  # no decay lr!
        )

        start_epoch = self.checkpoint_start_epoch if self.checkpoint_exists else 0

        # Iterate by epoch
        for epoch in range(start_epoch, EPOCHS):
            self.current_epoch = epoch

            if early_stopping.is_to_stop_training_early():
                break

            start = time.time()
            train_total_loss = 0.0
            val_total_loss = 0.0

            # Train by batch
            self.model.train()

            for batch_i, (imgs, targets) in enumerate(train_dataloader):

                train_loss = self.train_step(
                    imgs, targets
                )

                self._log_status("TRAIN", epoch, batch_i,
                                 train_dataloader, train_loss, print_freq)

                train_total_loss += train_loss

                # (only for debug: interrupt val after 1 step)
                if DISABLE_STEPS:
                    break

            # End training
            epoch_loss = train_total_loss/(batch_i+1)
            logging.info('Time taken for 1 epoch {:.4f} sec'.format(
                time.time() - start))
            logging.info('\n\n-----> TRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                                  train_total_loss/(batch_i+1)))

            # Start validation
            self.model.eval()  # eval mode (no dropout or batchnorm)

            with torch.no_grad():

                for batch_i, (imgs, targets) in enumerate(val_dataloader):

                    val_loss = self.val_step(
                        imgs, targets)

                    self._log_status("VAL", epoch, batch_i,
                                     val_dataloader, val_loss, print_freq)

                    val_total_loss += val_loss

                    # (only for debug: interrupt val after 1 step)
                    if DISABLE_STEPS:
                        break

            # End validation
            epoch_val_loss = val_total_loss/(batch_i+1)

            early_stopping.check_improvement(epoch_val_loss)

            self._save_checkpoint(early_stopping.is_current_val_best(),
                                  epoch,
                                  early_stopping.get_number_of_epochs_without_improvement(),
                                  epoch_val_loss)

            logging.info('\n-------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f} -------------\n'.format(
                epoch, EPOCHS, epoch_loss, epoch_val_loss))

    def _log_status(self, train_or_val, epoch, batch_i, dataloader, loss, print_freq):
        if batch_i % print_freq == 0:
            logging.info(
                "{} - Epoch: [{}/{}]; Batch: [{}/{}]\t Loss: {:.4f}\t".format(
                    train_or_val, epoch, EPOCHS, batch_i,
                    len(dataloader), loss
                )
            )

    def _save_checkpoint(self, val_loss_improved, epoch, epochs_since_last_improvement, val_loss):
        if val_loss_improved:

            state = {'epoch': epoch,
                     'epochs_since_last_improvement': epochs_since_last_improvement,
                     'val_loss': val_loss,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     }
            torch.save(state, self.get_checkpoint_path())

    def _load_weights_from_checkpoint(self, load_to_train):
        checkpoint_path = self.get_checkpoint_path()

        if os.path.exists(checkpoint_path):
            self.checkpoint_exists = True

            checkpoint = torch.load(checkpoint_path)

            # load model weights
            self.model.load_state_dict(checkpoint['model'])

            if load_to_train:
                # load optimizers and start epoch
                self.optimizer.load_state_dict(
                    checkpoint['optimizer'])

                self.checkpoint_start_epoch = checkpoint['epoch'] + 1
                self.checkpoint_epochs_since_last_improvement = checkpoint[
                    'epochs_since_last_improvement'] + 1
                self.checkpoint_val_loss = checkpoint['val_loss']

                logging.info(
                    "Restore model from checkpoint. Start epoch %s ", self.checkpoint_start_epoch)
        else:
            logging.info(
                "No checkpoint. Will start model from beggining\n")

    def get_checkpoint_path(self):
        path = self.MODEL_DIRECTORY + FILE_NAME+'.pth.tar'
        return path


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    classification_state = torch.load("src/data/RSICD/datasets/classification_dataset")
    classes_to_id = classification_state["classes_to_id"]
    id_to_classes = classification_state["id_to_classes"]
    classification_dataset = classification_state["classification_dataset"]

    dataset_len = len(classification_dataset)
    split_ratio = int(dataset_len * 0.10)

    classification_train = dict(list(classification_dataset.items())[split_ratio:])
    classification_val = dict(list(classification_dataset.items())[0:split_ratio])

    train_dataset_args = (classification_train, PATH_RSICD+"raw_dataset/RSICD_images/", classes_to_id)
    val_dataset_args = (classification_val, PATH_RSICD+"raw_dataset/RSICD_images/", classes_to_id)

    train_dataloader = DataLoader(
        ClassificationDataset(*train_dataset_args),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_dataloader = DataLoader(
        ClassificationDataset(*val_dataset_args),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    vocab_size = len(classes_to_id)

    model = ClassificationModel(vocab_size, device)
    model.setup_to_train()
    model.train(train_dataloader, val_dataloader)

    # checkpoint = torch.load('experiments/results/classification_finetune.pth.tar')

    # image_model = models.densenet201(pretrained=True)
    # num_features = image_model.classifier.in_features
    # image_model.classifier = nn.Linear(num_features, vocab_size)

    # image_model.load_state_dict(checkpoint['model'])

    # for batch, (img, target) in enumerate(train_dataloader):
    #     if batch == 5:
    #         result = image_model(img)
    #         output = torch.sigmoid(result)
    #         output = output > 0.5

    #         print("result", result)
    #         print("output", output)

    #         print("target", target)
    #         text = [id_to_classes[i] for i, value in enumerate(target[0]) if value == 1]
    #         output_text = [id_to_classes[i] for i, value in enumerate(output[0]) if value == True]
    #         print("target ext", text)
    #         print("output_text ext", output_text)

    #         break
