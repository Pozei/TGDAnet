import torch
import torch.nn as nn
from Transformer import SSTransformer
from InfoNCE import domain_adaptation_loss
from Text_Encode_Module import Text_Encode_Module


class TGDAnet(nn.Module):
    def __init__(self, 
                 batch_size,
                 patch_size,
                 src_dim,
                 src_class,
                 tar_dim,
                 tar_class,
                 features_dim,
                 device,
                 src_specialized_descriptions,
                 tar_specialized_descriptions,
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.features_dim = features_dim // 4
        dim = (src_dim + tar_dim) // 9

        ## Image Feature Extraction
        ### ZoomScales
        self.Zs = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(10)])

        ## 
        self.src_DR = nn.Sequential(
            nn.Conv2d(in_channels=src_dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))
        self.tar_DR = nn.Sequential(
            nn.Conv2d(in_channels=tar_dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

        ### Attention Modules for Source and Target
        self.src_CAM3D = CAM3D_Module()
        self.tar_CAM3D = CAM3D_Module()

        ### Spectral-Spatial Transformers for feature extraction
        self.TD1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

        self.transformer = SSTransformer(
                patch_sizes = patch_size, 
                in_channel = dim, 
                out_channel = self.features_dim, 
                depth= 9, 
                head = 6, 
                dim_head = 16, 
                mlp_head = 5,
                dropout = 0.6,)
        
        self.TD2 = nn.Sequential(
            nn.Conv2d(in_channels=self.features_dim, out_channels=self.features_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.features_dim),
            nn.ReLU(inplace=True))
        
        self.FT_DR = nn.Sequential(
            nn.Linear(in_features=self._get_layer_size(), out_features=features_dim*2, bias=False),
            nn.BatchNorm1d(features_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=features_dim*2, out_features=features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU(inplace=True))

        ### Prediction layer
        self.initial_predict = nn.Sequential(
            nn.Linear(in_features=features_dim,    out_features=features_dim//2),
            nn.LayerNorm(features_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=features_dim//2, out_features=src_class))
        
        ## Text Feature Extraction for Source and Target
        self.TEM = Text_Encode_Module(
            devices=device,
            src_specialized_descriptions=src_specialized_descriptions,
            tar_specialized_descriptions=tar_specialized_descriptions,
            features_dim=features_dim,         
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers)
        
        self.predict = nn.Sequential(
            nn.Linear(in_features=features_dim,    out_features=features_dim//4),
            nn.LayerNorm(features_dim//4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=features_dim//4, out_features=src_class))
        
    def FeatureExtractor(self, feature):
        feature = self.TD1(feature)
        feature = self.transformer(feature)
        feature = self.TD2(feature)
        feature = self.FT_DR(feature.flatten(start_dim=1))
        return feature


    def forward(self, src_time1, src_time2, tar_time1, tar_time2):

        ## Source Domain Feature Extraction
        src_dif_fea = src_time1 - self.Zs[0]*src_time2
        src_dif_fea = self.src_CAM3D(src_dif_fea)
        src_dif_fea = self.src_DR(src_dif_fea)
        src_dif_fea = torch.abs(src_dif_fea)

        src_feature = self.FeatureExtractor(src_dif_fea)
        src_predict = self.initial_predict(src_feature)
        src_label   = src_predict.data.max(1)[1]

        ## Target Domain Feature Extraction
        tar_dif_fea = tar_time1 - self.Zs[1]*tar_time2
        tar_dif_fea = self.tar_CAM3D(tar_dif_fea)
        tar_dif_fea = self.tar_DR(tar_dif_fea)
        tar_dif_fea = torch.abs(tar_dif_fea)

        tar_feature = self.FeatureExtractor(tar_dif_fea)
        tar_predict = self.initial_predict(tar_feature)
        tar_label   = tar_predict.data.max(1)[1]

        loss_DA1 = domain_adaptation_loss(src_feature, src_label, tar_feature, tar_label)

        ## Text Embedding Loss Calculation
        loss_txt, src_feature, tar_feature = self.TEM(src_feature, src_label, tar_feature, tar_label)

        src_predict = self.predict(src_feature)
        src_label   = src_predict.data.max(1)[1]
        tar_predict = self.predict(tar_feature)
        tar_label   = tar_predict.data.max(1)[1]

        loss_DA2 = domain_adaptation_loss(src_feature, src_label, tar_feature, tar_label)

        return src_predict, loss_DA1+loss_DA2, loss_txt

        
    def Source(self, src_time1, src_time2):

        ## Source Domain Feature Extraction
        src_dif_fea = src_time1 - self.Zs[0]*src_time2
        src_dif_fea = self.src_CAM3D(src_dif_fea)
        src_dif_fea = self.src_DR(src_dif_fea)
        src_dif_fea = torch.abs(src_dif_fea)

        src_feature = self.FeatureExtractor(src_dif_fea)
        src_predict = self.initial_predict(src_feature)
        src_label   = src_predict.data.max(1)[1]

        ## Text Embedding Loss Calculation
        src_feature = self.TEM.Source(src_feature, src_label)
        src_predict = self.predict(src_feature)

        return src_predict


    def Target(self, tar_time1, tar_time2):

        ## Target Domain Feature Extraction
        tar_dif_fea = tar_time1 - self.Zs[1]*tar_time2
        tar_dif_fea = self.tar_CAM3D(tar_dif_fea)
        tar_dif_fea = self.tar_DR(tar_dif_fea)
        tar_dif_fea = torch.abs(tar_dif_fea)

        tar_feature = self.FeatureExtractor(tar_dif_fea)
        tar_predict = self.initial_predict(tar_feature)
        tar_label   = tar_predict.data.max(1)[1]
        
        ## Text Embedding Loss Calculation
        tar_feature = self.TEM.Target(tar_feature, tar_label)
        tar_predict = self.predict(tar_feature)

        return tar_predict


    def _get_layer_size(self):
        with torch.no_grad():

            # 使用动态输入计算输出维度
            out = torch.zeros((self.batch_size, self.features_dim, self.patch_size, self.patch_size))
            out = self.TD2(out)
            out = out.view(out.size(0), -1)

            
        return out.size(1)
    



class CAM3D_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM3D_Module, self).__init__()

        self.gamma   = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x  : input feature maps(B X C X H X W)
            returns:
                out: attention value + input feature
                attention: B X C X C
        """
        batchsize, Channel, height, width = x.shape

        CAM_query = x.view(batchsize, Channel, -1)
        CAM_key   = x.view(batchsize, Channel, -1).permute(0, 2, 1)
        CAM_value = x.view(batchsize, Channel, -1)

        CAM_energy = torch.bmm(CAM_query, CAM_key)  ## matrix product 
        CAM_energy = torch.max(CAM_energy, -1, keepdim=True)[0].expand_as(CAM_energy) - CAM_energy #其中通过减去最大值可以避免指数运算溢出，并且有助于捕捉到更多的细微差异。
        CAM_attention  = self.softmax(CAM_energy) ## Normalize
        
        out = torch.bmm(CAM_attention , CAM_value)
        out = self.softmax(out) 
        out = out.view(batchsize, Channel, height, width)

        out = x + self.gamma*out
        
        return out