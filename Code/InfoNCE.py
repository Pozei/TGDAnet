import torch
import torch.nn.functional as F

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
             
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
        
    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ negative_keys.t()

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ negative_keys.t()
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ positive_key.t()

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)



def contrastive_loss(feature1, feature2):

    if feature1.dim() != 2:
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
    else:
        pass
    
    Loss_contrastive = (info_nce(query=feature1, positive_key=feature1, negative_keys=feature2) + 
                        info_nce(query=feature2, positive_key=feature2, negative_keys=feature1))/2
    
    return Loss_contrastive


def domain_adaptation_loss(src_feature, src_label, tar_feature, tar_label):

    loss_global = contrastive_loss(src_feature,tar_feature)
    loss_local_list = []
    list_C = torch.unique(src_label&tar_label)
    for c in list_C:

        src_pos_position = torch.where(src_label == c)[0]
        tar_pos_position = torch.where(tar_label == c)[0]
        src_pos_feature  = src_feature[src_pos_position]
        tar_pos_feature  = tar_feature[tar_pos_position]
        loss_local_c = contrastive_loss(src_pos_feature, tar_pos_feature)
        loss_local_list.append(loss_local_c)
    loss_local = torch.sum(torch.stack(loss_local_list))


    return loss_global+loss_local
   

