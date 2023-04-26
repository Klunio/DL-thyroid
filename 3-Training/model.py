import datetime

import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def create_model(num_layers, pretrain):
    assert num_layers in [18, 34, 50, 101, 152]
    architecture = "resnet{}".format(num_layers)
    model_constructor = getattr(torchvision.models, architecture)
    model = model_constructor(num_classes=2)

    if pretrain is True:
        pretrained = model_constructor(pretrained=True).state_dict()
        if pretrained['fc.weight'].size(0) != 2:
            del pretrained['fc.weight'], pretrained['fc.bias']
        model.load_state_dict(pretrained, strict=False)
    return model


def resnet_extractor(num_layer, pretrain):
    resnet_feature_extractor = create_model(num_layer, pretrain)

    if num_layer in [18, 34]:
        resnet_feature_extractor.fc = nn.Linear(512, 512)
    elif num_layer in [50, 101]:
        resnet_feature_extractor.fc = nn.Linear(2048, 1024)

    nn.init.xavier_normal_(resnet_feature_extractor.fc.weight)
    return resnet_feature_extractor


def inception_extractor(pretrain):
    inception_feature_extractor = torchvision.models.inception_v3(
        pretrain, aux_logits=False)

    inception_feature_extractor.fc = nn.Linear(2048, 1024)
    nn.init.xavier_normal_(inception_feature_extractor.fc.weight)

    return inception_feature_extractor


def inception_v4_extractor(pretrain):
    inception_feature_extractor = pretrainedmodels.inceptionv4(
        pretrained='imagenet')

    inception_feature_extractor.last_linear = nn.Linear(1536, 512)

    nn.init.xavier_normal_(inception_feature_extractor.last_linear.weight)

    return inception_feature_extractor


def inception_resnet_extractor(pretrain):
    inception_feature_extractor = pretrainedmodels.inceptionresnetv2(
        pretrained='imagenet')

    inception_feature_extractor.last_linear = nn.Sequential(
        nn.Linear(1536, 1024),
        nn.ReLU()
    )
    nn.init.xavier_normal_(inception_feature_extractor.last_linear[0].weight)

    return inception_feature_extractor


def nasnet_extractor(pretrain):
    nasnet_feature_extractor = pretrainedmodels.nasnetamobile(num_classes=1000)

    nasnet_feature_extractor.last_linear = nn.Linear(1056, 1024)
    nn.init.xavier_normal_(
        nasnet_feature_extractor.last_linear.weight
    )

    return nasnet_feature_extractor


def nasnet_large_extractor(pretrain):
    nasnet_feature_extractor = pretrainedmodels.nasnetalarge(num_classes=1000)

    nasnet_feature_extractor.last_linear = nn.Linear(4032, 1024)
    nn.init.xavier_normal_(
        nasnet_feature_extractor.last_linear.weight
    )
    return nasnet_feature_extractor


def pnasnet_large_extractor(pretrain):
    pnasnet_feature_extractor = pretrainedmodels.pnasnet5large(
        num_classes=1000)

    pnasnet_feature_extractor.last_linear = nn.Linear(4320, 1024)
    nn.init.xavier_normal_(
        pnasnet_feature_extractor.last_linear.weight
    )
    return pnasnet_feature_extractor


def mobilenetv3_extractor(pretrain):
    import mobilenetv3
    mobilenetv3_feature_extractor = mobilenetv3.mobilenetv3_large()
    mobilenetv3_feature_extractor.load_state_dict(
        torch.load('./pretrained/mobilenetv3-large-1cd25616.pth')
    )
    mobilenetv3_feature_extractor.classifier[3] = nn.Linear(1280, 1024)
    nn.init.xavier_normal_(
        mobilenetv3_feature_extractor.classifier[3].weight
    )

    return mobilenetv3_feature_extractor


feature_extractor_dict = {
    'inceptionv3': inception_extractor,
    'inceptionv4': inception_v4_extractor,
    'inception-resnet': inception_resnet_extractor,
    'nasnet': nasnet_extractor,
    'nasnetalarge': nasnet_large_extractor,
    'mobilenetv3': mobilenetv3_extractor,
    'pnasnet': pnasnet_large_extractor
}


class Attention(nn.Module):
    def __init__(self, model='resnet-18', pretrain=True):
        super(Attention, self).__init__()

        if model in ['resnet-18', 'resnet-34']:
            self.M = 512
            self.L = 320
            self.D = 128
        elif model in ['inceptionv3', 'inceptionv4', 'resnet-50', 'inception-resnet', 'nasnet', 'mobilenetv3',
                       'nasnetalarge', 'pnasnet']:
            self.M = 512
            self.L = 256
            self.D = 128
        self.K = 1

        self.feature_extractor_part1 = feature_extractor_dict[model](pretrain)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_normal_(self.feature_extractor_part2[0].weight)
        nn.init.xavier_normal_(self.attention[0].weight)
        nn.init.xavier_normal_(self.attention[2].weight)
        nn.init.xavier_normal_(self.classifier[0].weight)

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)  # N x M
        H = H.view(-1, self.M)  # N x M
        H = self.feature_extractor_part2(H)  # N x L

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob


class Attention_Test(Attention):
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.M)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK

        return H, A


class Attention_Gated(Attention):
    def __init__(self, model, pretrain):
        super(Attention_Gated, self).__init__(model, pretrain)
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        nn.init.xavier_normal_(self.attention_V[0].weight)

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.attention_U[0].weight)

        self.attention = nn.Linear(self.D, self.K)
        nn.init.xavier_normal_(self.attention.weight)
        self.g = torch.Generator()
        self.g.manual_seed(hash(datetime.datetime.now()))

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)  # N x M
        H = H.view(-1, self.M)  # N x M
        H = self.feature_extractor_part2(H)  # N x L

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention(A_V * A_U)  # NxK

        idx = torch.randperm(A.shape[0])[:32]
        A, H = A[idx], H[idx]

        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob
