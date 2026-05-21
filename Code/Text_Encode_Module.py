import clip
import torch
import torch.nn as nn
from InfoNCE import domain_adaptation_loss, contrastive_loss

class Text_Encode_Module(nn.Module):
    def __init__(self, 
                 devices,
                 src_specialized_descriptions, 
                 tar_specialized_descriptions,  
                 features_dim,               
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers):
        super().__init__()

        self.device = devices
        self.label_names = ["no change", "change"]

        self.generalized_descriptions = {
            "no change": ["This patch of the hyperspectral image shows no change. This indicates that the features in the multitemporal images are either only slightly different or show no difference."],
            "change": ["This patch of the hyperspectral image shows change. This indicates that there are significant differences between the multitemporal images."]}
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.src_specialized_descriptions = src_specialized_descriptions
        self.tar_specialized_descriptions = tar_specialized_descriptions

        self.LayerNorm = nn.LayerNorm(transformer_width)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  
        self.text_projection = nn.Parameter(torch.empty(transformer_width, features_dim))                    
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=0.01)

        self.transformer = Transformer(
            width=transformer_width,
            heads=transformer_heads,
            layers=transformer_layers,
            attn_mask=self.build_attention_mask(),
            device=devices)
        
        ### ZoomScales
        self.Zs = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(10)])

        self.DR0 = nn.Sequential(
                nn.Linear(in_features=features_dim, out_features=features_dim),
                nn.BatchNorm1d(features_dim),
                nn.ReLU(inplace=True))
        self.DR1 = nn.Sequential(
                nn.Linear(in_features=features_dim, out_features=features_dim),
                nn.BatchNorm1d(features_dim),
                nn.ReLU(inplace=True))
        self.DR2 = nn.Sequential(
                nn.Linear(in_features=features_dim, out_features=features_dim),
                nn.BatchNorm1d(features_dim),
                nn.ReLU(inplace=True))
        self.DR3 = nn.Sequential(
                nn.Linear(in_features=features_dim, out_features=features_dim),
                nn.BatchNorm1d(features_dim),
                nn.ReLU(inplace=True))
        
        self.src_token = nn.Parameter(torch.randn(1, features_dim))
        self.tar_token = nn.Parameter(torch.randn(1, features_dim))

    def forward(self, src_img_fea, src_label, tar_img_fea, tar_label):
         
        device = self.device
        label_name = self.label_names

        # ---------------------- 源域文本特征提取 ---------------------- #
        src_generalized_descriptions = self.generalized_descriptions
        src_generalized = [src_generalized_descriptions[label_name[k]][0] for k in src_label]
        src_generalized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in src_generalized]))
        src_generalized = src_generalized / src_generalized.norm(dim=1, keepdim=True)

        src_specialized_descriptions = self.src_specialized_descriptions
        src_specialized = [src_specialized_descriptions[label_name[k]][0] for k in src_label]
        src_specialized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in src_specialized]))
        src_specialized = src_specialized / src_specialized.norm(dim=1, keepdim=True)

        # ---------------------- 目标域文本特征提取 ---------------------- #
        tar_generalized_descriptions = self.generalized_descriptions
        tar_generalized = [tar_generalized_descriptions[label_name[k]][0] for k in tar_label]
        tar_generalized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in tar_generalized]))
        tar_generalized = tar_generalized / tar_generalized.norm(dim=1, keepdim=True)

        tar_specialized_descriptions = self.tar_specialized_descriptions
        tar_specialized = [tar_specialized_descriptions[label_name[k]][0] for k in tar_label]
        tar_specialized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in tar_specialized]))
        tar_specialized = tar_specialized / tar_specialized.norm(dim=1, keepdim=True)

        # ---------------------- 文本与图像特征融合及对齐 ---------------------- #
        src_common = self.DR0(src_img_fea - self.Zs[0]*src_specialized)
        tar_common = self.DR0(tar_img_fea - self.Zs[1]*tar_specialized)

        loss_common = (domain_adaptation_loss(src_common,      src_label, tar_generalized, tar_label) + 
                       domain_adaptation_loss(src_generalized, src_label,  tar_common, tar_label))

        src_special = self.DR2(src_img_fea - self.Zs[4]*src_generalized)
        tar_special = self.DR3(tar_img_fea - self.Zs[5]*tar_generalized)

        loss_special= (contrastive_loss(src_special, src_specialized) + contrastive_loss(tar_special, tar_specialized))

        src_feature = src_common + self.Zs[2]*src_generalized
        tar_feature = tar_common + self.Zs[3]*tar_generalized


        return loss_common+loss_special, src_feature, tar_feature


    def Source(self, src_img_fea, src_label):
         
        device = self.device
        label_name = self.label_names

        # ---------------------- 源域文本特征提取 ---------------------- #
        src_generalized_descriptions = self.generalized_descriptions
        src_generalized = [src_generalized_descriptions[label_name[k]][0] for k in src_label]
        src_generalized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in src_generalized]))
        src_generalized = src_generalized / src_generalized.norm(dim=1, keepdim=True)

        src_specialized_descriptions = self.src_specialized_descriptions
        src_specialized = [src_specialized_descriptions[label_name[k]][0] for k in src_label]
        src_specialized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in src_specialized]))
        src_specialized = src_specialized / src_specialized.norm(dim=1, keepdim=True)

        # ---------------------- 文本与图像特征融合及对齐 ---------------------- #
        src_common  = self.DR0(src_img_fea - self.Zs[0]*src_specialized)
        src_feature = src_common  + self.Zs[2]*src_generalized


        return src_feature


    def Target(self,tar_img_fea, tar_label):
         
        device = self.device
        label_name = self.label_names

        # ---------------------- 目标域文本特征提取 ---------------------- #
        tar_generalized_descriptions = self.generalized_descriptions
        tar_generalized = [tar_generalized_descriptions[label_name[k]][0] for k in tar_label]
        tar_generalized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in tar_generalized]))
        tar_generalized = tar_generalized / tar_generalized.norm(dim=1, keepdim=True)

        tar_specialized_descriptions = self.tar_specialized_descriptions
        tar_specialized = [tar_specialized_descriptions[label_name[k]][0] for k in tar_label]
        tar_specialized = self.encode_text(torch.cat([clip.tokenize(k).to(device) for k in tar_specialized]))
        tar_specialized = tar_specialized / tar_specialized.norm(dim=1, keepdim=True)

        # ---------------------- 文本与图像特征融合及对齐 ---------------------- #
        tar_common  = self.DR0(tar_img_fea - self.Zs[1]*tar_specialized)
        tar_feature = tar_common + self.Zs[3]*tar_generalized


        return tar_feature




    def encode_text(self, text):
        # Text embedding
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.LayerNorm(x)

        # Project final text feature
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        return mask





class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor, device: torch.device):
        super().__init__()
        self.layers = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model))
        
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)