import os
import time
import torch
import argparse
import numpy as np
import scipy.io as sio
from TGDAnet import TGDAnet
from sklearn import metrics
import torch.optim as optim
from GeneratePic import generate_png, ReshapeLabel
import torch.backends.cudnn as cudnn
from datasets_utils import get_dataset, gain_train_valid_batch, gain_total_batch


def TGDAnet_main():

    # 1. Configuration parameters
    parser = argparse.ArgumentParser(description="TGDAnet Change Detection")

    ## 1.1 Data parameters
    parser.add_argument('--path',    type=str,   default='Code/',     help="The root path of the project.")
    parser.add_argument('--folder',  type=str,   default='Data/',     help='The path where the dataset is stored')
    parser.add_argument('--method',  type=str,   default='TGDAnet',   help='The name of the method.')
    parser.add_argument('--source',  type=str,   default='Bay',       help='The name of the source domain dataset.')
    parser.add_argument('--target',  type=str,   default='Santa',     help='The name of the target domain dataset.')

    ## 1.2 Training parameters
    parser.add_argument('--tr_rate',  type=str,  default=2.5,         help='The proportion of samples in training.'  )
    parser.add_argument('--val_rate', type=str,  default=10.0,        help='The proportion of samples in valid.' )
    parser.add_argument('--epoches', type=int,   default=50,          help='The number of training epoches.')
    parser.add_argument('--batches', type=int,   default=256,         help='The size of batches.')
    parser.add_argument('--patches', type=int,   default=11,          help='The size of patches.')
    parser.add_argument('--lr_rate', type=float, default=1e-4,        help='Learning rate')
    parser.add_argument('--w_deacy', type=float, default=8e-1,        help='Weight decay')
    parser.add_argument('--lambda1', type=float, default=1e-2,        help='The loss weighting for domain adaptation.')
    parser.add_argument('--lambda2', type=float, default=1e-2,        help='The loss weighting for text.')

    ## 1.3 System parameters
    parser.add_argument('--cuda',    type=str, default='2',           help='CUDA Device No.')
    parser.add_argument('--seed',    type=int, default=1024,          help='Random seed')
    args = parser.parse_args()
    
    ## 1.4 random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    ## 1.5 set up the PyTorch device
    device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')


    #  2.  Load dataset
    ## 2.1 Load source
    print(f'===== {args.source} Transform to {args.target} =====')

    print(f'    seed: {args.seed}; lambda1: {args.lambda1}; lambda2: {args.lambda2};')

    src_time1, src_time2, src_gt, src_specialized_descriptions = get_dataset(args.source, args.folder)
    _, _, src_dim = src_time1.shape
    src_class = src_gt.max()
    print(f'    Source: {args.source};  Source shape: {src_time1.shape};  Source class: {src_class};')
    src_train_loader, src_valid_loader = gain_train_valid_batch(src_time1, src_time2, src_gt, args)

    ## 2.2 Load target
    tar_time1, tar_time2, tar_gt, tar_specialized_descriptions = get_dataset(args.target, args.folder)
    _, _, tar_dim = tar_time1.shape
    tar_class = tar_gt.max()
    print(f'    Target: {args.target};  Target shape: {tar_time1.shape};  Target class: {tar_class};')
    tar_train_loader, _, = gain_train_valid_batch(tar_time1, tar_time2, tar_gt, args)


    # 3. Load model
    pretrained_dict   = torch.jit.load(args.path+'ViT-B-32.pt', map_location="cpu").state_dict()
    features_embed_dim= pretrained_dict ["text_projection"].shape[1]
    context_length    = pretrained_dict ["positional_embedding"].shape[0]
    vocab_size        = pretrained_dict ["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict ["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 128
    transformer_layers= 3

    model = TGDAnet(
        ## Image
        args.batches,
        args.patches,
        src_dim,
        src_class,
        tar_dim,
        tar_class,
        features_embed_dim,
        device,
        ## Text
        src_specialized_descriptions,
        tar_specialized_descriptions,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers).to(device)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict_filtered = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and 'visual' not in k.split('.'):
            pretrained_dict_filtered[k] = v
    pretrained_dict = pretrained_dict_filtered
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_rate, weight_decay=args.w_deacy)


    # 4. Train
    print(f'===== Train =====')
    best = 0
    for epoch in range(args.epoches):
        model.train()
        tar_train_iter = iter(tar_train_loader)  # Target domain data iterator
        Src_Train_Loss, Src_Train_GT, Src_Train_Pred = [], [], []
        for src_train_time1, src_train_time2, src_train_gt in src_train_loader:
            ## 4.1 Obtain target domain data
            try:
                tar_train_time1, tar_train_time2, _ = next(tar_train_iter)
            except StopIteration:
                tar_train_time1, tar_train_time2, _ = next(iter(tar_train_loader))

            ## 4.2 Move to CUDA
            src_train_time1, src_train_time2 = src_train_time1.to(device), src_train_time2.to(device)
            tar_train_time1, tar_train_time2 = tar_train_time1.to(device), tar_train_time2.to(device)

            ## 4.3  Forward propagation
            src_train_pred, loss_DA, loss_txt = model(src_train_time1, src_train_time2, tar_train_time1, tar_train_time2)

            loss_train = (criterion(src_train_pred, src_train_gt.to(device)) + args.lambda1*loss_DA + args.lambda2*loss_txt)

            ## 4.4 Backward propagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            ## 4.5 Cumulative results
            Src_Train_GT.extend(src_train_gt.cpu().numpy())
            Src_Train_Loss.append(loss_train.item())
            Src_Train_Pred.extend(src_train_pred.argmax(dim=1).cpu().numpy())
        
        ## 4.6 Calculate and print training indicators
        src_train_IoU= metrics.jaccard_score(Src_Train_GT, Src_Train_Pred)*100
        src_train_Ka = metrics.cohen_kappa_score(Src_Train_GT, Src_Train_Pred)
        

    # 5.Valid
        if (epoch + 1) % 10 == 0:
            model.eval()
            Src_Valid_Loss, Src_Valid_GT, Src_Valid_Pred = [], [], []
            with torch.no_grad():
                for src_valid_time1, src_valid_time2, src_valid_gt in src_valid_loader:
                    ## 5.1 Move to CUDA
                    src_valid_time1, src_valid_time2 = (src_valid_time1).to(device), (src_valid_time2).to(device)

                    src_valid_pred = model.Source(src_valid_time1, src_valid_time2)

                    loss_valid = criterion(src_valid_pred, src_valid_gt.to(device))

                    ## 5.1 Cumulative results
                    Src_Valid_GT.extend(src_valid_gt.cpu().numpy())
                    Src_Valid_Loss.append(loss_valid.item())
                    Src_Valid_Pred.extend(src_valid_pred.argmax(dim=1).cpu().numpy())

                src_valid_IoU= metrics.jaccard_score(Src_Valid_GT, Src_Valid_Pred)*100
                src_valid_Ka = metrics.cohen_kappa_score(Src_Valid_GT, Src_Valid_Pred)

                ## 5.2 Preservation of the best models
                if src_valid_IoU >= best:
                    best = src_valid_IoU
                    name_pt = f'{args.path}{args.method}_pt_{args.source}2{args.target}.pt'
                    torch.save(model, name_pt)

                ## 5.3 Print the training log
                print(f'    Epoch: {epoch + 1}/{args.epoches}; '
                    f'Src_Train Loss: {np.mean(Src_Train_Loss):.4f}; Src_Train IoU: {src_train_IoU:.2f}; Src_Train Ka: {src_train_Ka:.4f}; '
                    f'Src_Valid Loss: {np.mean(Src_Valid_Loss):.4f}; Src_Valid IoU: {src_valid_IoU:.2f}; Src_Valid Ka: {src_valid_Ka:.4f};')
   
   
    # 6. Test
    print(f'===== Test =====')
    tar_loader = gain_total_batch(tar_time1, tar_time2, tar_gt, args)

    name_pt = f'{args.path}{args.method}_pt_{args.source}2{args.target}.pt'
    model = torch.load(name_pt)
    model.eval()
    Tar_Position, Tar_GT, Tar_Pre_label = [], [], []
    with torch.no_grad():
        for tar_total_position, tar_total_time1, tar_total_time2, tar_total_gt in tar_loader:
            ## 6.1 Move to CUDA
            tar_total_time1, tar_total_time2 = tar_total_time1.to(device), tar_total_time2.to(device)

            tar_total_pred = model.Target(tar_total_time1, tar_total_time2)

            ## 6.2 Cumulative results
            Tar_GT.extend(tar_total_gt.cpu().numpy())
            Tar_Pre_label.extend(tar_total_pred.argmax(dim=1).cpu().numpy())      
            Tar_Position.extend(tar_total_position.cpu().numpy())

    Tar_GT       = np.array(Tar_GT)
    Tar_Pre_label= np.array(Tar_Pre_label)
    Tar_Position = np.array(Tar_Position) 

    metrics_report = {
        'IoU': metrics.jaccard_score(Tar_GT, Tar_Pre_label),
        'F1': metrics.f1_score(Tar_GT, Tar_Pre_label),
        'Kappa': metrics.cohen_kappa_score(Tar_GT, Tar_Pre_label),
        'OA': metrics.accuracy_score(Tar_GT, Tar_Pre_label),
        'Precision': metrics.precision_score(Tar_GT, Tar_Pre_label),
        'Recall': metrics.recall_score(Tar_GT, Tar_Pre_label)}
    

    # 7. Reshaping Predictive Labeling
    tar_label = ReshapeLabel(tar_gt, Tar_Pre_label, Tar_Position)


    return metrics_report, tar_gt, tar_label, args



if __name__ == "__main__": 

    # 1. Run model
    print(f'\n=============== Start ===============')  
    start_time = time.time()

    metrics_report, Tar_GT, Tar_Predict, args = TGDAnet_main()

    duration = time.time() - start_time

    print(f'===== The Final Accuracy for Evaluation. =====')
    print(f'    Time: {duration:.2f}s')
    print(f"    IoU: {metrics_report['IoU'] * 100:.2f}% | "
        f"F1: {metrics_report['F1'] * 100:.2f}% | "
        f"OA: {metrics_report['OA'] * 100:.2f}% | "
        f"Kappa: {metrics_report['Kappa']:.4f} | "
        f"Precision: {metrics_report['Precision']:.4f} | "
        f"Recall: {metrics_report['Recall']:.4f}")
    
    # 2. 
    path_name = os.path.join(args.path, 'result')
    os.makedirs(path_name, exist_ok=True)
    sio.savemat(os.path.join(path_name, f'MAT_{args.source}2{args.target}.mat'), {f'{args.source}2{args.target}': Tar_Predict})

    # 2. Generate result PNG
    _ = generate_png(Tar_GT, Tar_Predict, args)

    print(f'=============== END ===============\n')


"""

=============== Start ===============
===== Bay Transform to Santa =====
    seed: 1024; lambda1: 0.01; lambda2: 0.01;
    Source: Bay;  Source shape: (600, 500, 224);  Source class: 2;
    Train samples: 1837;  Valid samples: 7348
    Target: Santa;  Target shape: (984, 740, 224);  Target class: 2;
    Train samples: 3313;  Valid samples: 13255
===== Train =====
    Epoch: 10/50; Src_Train Loss: 0.0769; Src_Train IoU: 95.85; Src_Train Ka: 0.9539; Src_Valid Loss: 0.1016; Src_Valid IoU: 95.30; Src_Valid Ka: 0.9494;
    Epoch: 20/50; Src_Train Loss: 0.0448; Src_Train IoU: 97.22; Src_Train Ka: 0.9697; Src_Valid Loss: 0.0872; Src_Valid IoU: 96.19; Src_Valid Ka: 0.9589;
    Epoch: 30/50; Src_Train Loss: 0.0611; Src_Train IoU: 96.17; Src_Train Ka: 0.9586; Src_Valid Loss: 0.0845; Src_Valid IoU: 96.70; Src_Valid Ka: 0.9641;
    Epoch: 40/50; Src_Train Loss: 0.0344; Src_Train IoU: 98.44; Src_Train Ka: 0.9832; Src_Valid Loss: 0.1145; Src_Valid IoU: 94.99; Src_Valid Ka: 0.9464;
    Epoch: 50/50; Src_Train Loss: 0.0202; Src_Train IoU: 99.16; Src_Train Ka: 0.9910; Src_Valid Loss: 0.0587; Src_Valid IoU: 97.54; Src_Valid Ka: 0.9734;
===== Test =====
    Total samples:  132552
===== The Final Accuracy for Evaluation. =====
    Time: 1442.02s
    IoU: 84.53% | F1: 91.62% | OA: 93.16% | Kappa: 0.8586 | Precision: 0.8846 | Recall: 0.9501
    Saving Ground Truth image...
    Saving Binary Prediction image...
    Saving Multi-class Prediction image...
=============== END ===============

"""













