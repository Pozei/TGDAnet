import torch
import random
import numpy as np
import torch.utils
import scipy.io as sio
import torch.utils.data as Data
import scipy.ndimage as ndimage
from sklearn import preprocessing

def seed_worker(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataset(dataset_name, folder):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """

    if dataset_name == 'China':
        # Load the image
        Time1 = sio.loadmat(folder +'/01_China/China_20060503.mat')['China_20060503']
        Time2 = sio.loadmat(folder +'/01_China/China_20070423.mat')['China_20070423']
        Bi_GT = sio.loadmat(folder +'/01_China/China_GT.mat')['China_GT']
        specialized_descriptions = {
            "no change":['The unchanged portion of the dataset represents areas that remain consistent throughout the crop rotation process.'],
            "change":['The changes observed in this dataset were primarily due to crop rotation. ']}
    
    elif dataset_name == 'Benton':
        # Load the image
        Time1 = sio.loadmat(folder +'/02_Benton/Benton_2004.mat')['Benton_2004']
        Time2 = sio.loadmat(folder +'/02_Benton/Benton_2007.mat')['Benton_2007']
        Bi_GT = sio.loadmat(folder +'/02_Benton/Benton_GT.mat')['Benton_GT']
        specialized_descriptions = {
        "no change":['The unchanged portion of the dataset represents areas that remain consistent throughout crop transitions.'],
        "change":['The changes observed in this dataset are significant for crop transitions.']}

    elif dataset_name == 'USA':
        # Load the image
        Time1 = sio.loadmat(folder +'/03_USA/USA_20040501.mat')['USA_20040501']
        Time2 = sio.loadmat(folder +'/03_USA/USA_20070508.mat')['USA_20070508']
        Bi_GT = sio.loadmat(folder +'/03_USA/USA_GT.mat')['USA_GT']
        specialized_descriptions = {
        "no change":['The unchanged portion of the dataset corresponds to areas that remain unaffected by the irrigation regulation.'],
        "change":['The changes observed in this dataset are primarily attributed to the regulation of the irrigated area.']}

    elif dataset_name == 'River':
        # Load the image
        Time1 = sio.loadmat(folder +'/04_River/River_20230503.mat')['River_20230503']
        Time2 = sio.loadmat(folder +'/04_River/River_20231231.mat')['River_20231231']
        Bi_GT = sio.loadmat(folder +'/04_River/River_GT.mat')['River_GT']
        specialized_descriptions = {
        "no change":['The unchanged portion of the dataset represents the area that is unaffected by changes in the material within the river.'],
        "change":['The changes observed in this dataset are primarily due to the removal of material from the river.']}

    elif dataset_name == 'Santa':
        # Load the image
        Time1 = sio.loadmat(folder +'/05_SantaBarbara/SantaBarbara_2013.mat')['SantaBarbara_2013']
        Time2 = sio.loadmat(folder +'/05_SantaBarbara/SantaBarbara_2014.mat')['SantaBarbara_2014']
        Bi_GT = sio.loadmat(folder +'/05_SantaBarbara/SantaBarbara_GT.mat')['SantaBarbara_GT']
        specialized_descriptions = {
        "no change":['The unchanged portion of the dataset represents areas that have remained stable during the evolution of the city and changes in agricultural land use.'],
        "change":['The changes observed in this dataset reflect the evolution of the city and the dynamics of agricultural land.']}

    elif dataset_name == 'Bay':
        # Load the image
        Time1 = sio.loadmat(folder +'/06_BayArea/BayArea_2013.mat')['BayArea_2013']
        Time2 = sio.loadmat(folder +'/06_BayArea/BayArea_2015.mat')['BayArea_2015']
        Bi_GT = sio.loadmat(folder +'/06_BayArea/BayArea_GT.mat')['BayArea_GT']
        specialized_descriptions = {
        "no change":['The unchanged portions of the dataset represent areas that are unaffected by changes in farmlands and buildings.'],
        "change":['The changes observed in this dataset are primarily associated with farmlands and buildings.']}


    # Normalization
    h, w, d = Time1.shape
    Time1_temp = Time1.reshape(h*w, d)
    Time2_temp = Time2.reshape(h*w, d)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    Time1_temp = scaler.fit_transform(Time1_temp)
    Time2_temp = scaler.fit_transform(Time2_temp)
    Time1 = Time1_temp.reshape(h, w, d)
    Time2 = Time2_temp.reshape(h, w, d)


    return Time1, Time2, Bi_GT,  specialized_descriptions


def sample_position(GT, train_rate, valid_rate,seed):

    seed_worker(seed)
    train_position_list, valid_position_list = [], []
    for i in range(2):
        Position_i = np.argwhere(GT == i)
            
        num_samples = Position_i.shape[0]
        random_indices = np.random.permutation(num_samples)
            
        num_train_i = max(round(num_samples * train_rate), 1)
        num_valid_i = max(round(num_samples * valid_rate), 1)

        train_indices = random_indices[:num_train_i]
        train_position_list.append(Position_i[train_indices])

        valid_indices = random_indices[num_train_i:num_train_i+num_valid_i]
        valid_position_list.append(Position_i[valid_indices])

    train_position = np.vstack(train_position_list)
    valid_position = np.vstack(valid_position_list)
        
    return train_position, valid_position
    
def gain_patch(image, x, y, patch_sizes):
    return image[x:(x+patch_sizes), y:(y+patch_sizes), :]
    
def gain_train_valid_batch(Time1, Time2, GT, args):

    H, W, Dim = Time1.shape
    r = args.patches // 2
    Time1_mirror = np.pad(Time1, ((r,r), (r,r), (0,0)), mode='reflect')
    Time2_mirror = np.pad(Time2, ((r,r), (r,r), (0,0)), mode='reflect')

    train_position, valid_position = sample_position(GT, args.tr_rate/100, args.val_rate/100 ,args.seed)
    print(f'    Train samples: {train_position.shape[0]};  Valid samples: {valid_position.shape[0]}')

    ## Train Samples
    train_T1 = np.zeros((train_position.shape[0], args.patches, args.patches, Dim))
    train_T2 = np.zeros((train_position.shape[0], args.patches, args.patches, Dim))
    train_gt = np.zeros((train_position.shape[0],))
    for i in range(train_position.shape[0]):
        temp_x, temp_y = train_position[i,0], train_position[i,1]
        train_T1[i,:,:,:] = gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=args.patches)
        train_T2[i,:,:,:] = gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=args.patches)
        train_gt[i]       = GT[temp_x, temp_y]

    T1_train = torch.from_numpy(train_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_train = torch.from_numpy(train_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_train  = torch.from_numpy(train_gt).type(torch.LongTensor)

    batch_train = Data.TensorDataset(T1_train, T2_train, Y_train)
    Train_loader = Data.DataLoader(dataset=batch_train,
                                    pin_memory=True,
                                    worker_init_fn=lambda worker_id: seed_worker(args.seed),
                                    batch_size=args.batches,
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=6,)
        
    ## Valid Samples
    valid_T1 = np.zeros((valid_position.shape[0], args.patches, args.patches, Dim))
    valid_T2 = np.zeros((valid_position.shape[0], args.patches, args.patches, Dim))
    valid_gt = np.zeros((valid_position.shape[0],))
    for i in range(valid_position.shape[0]):
        temp_x, temp_y = valid_position[i,0], valid_position[i,1]
        valid_T1[i,:,:,:] = gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=args.patches)
        valid_T2[i,:,:,:] = gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=args.patches)
        valid_gt[i]       = GT[temp_x, temp_y]

    T1_test = torch.from_numpy(valid_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_test = torch.from_numpy(valid_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_test  = torch.from_numpy(valid_gt).type(torch.LongTensor)

    batch_valid   = Data.TensorDataset(T1_test, T2_test, Y_test)
    Valid_loader = Data.DataLoader(dataset=batch_valid,
                                    pin_memory=True,
                                    worker_init_fn=lambda worker_id: seed_worker(args.seed),
                                    batch_size=args.batches,
                                    drop_last=True,
                                    shuffle=True,
                                    num_workers=6,)

    return Train_loader, Valid_loader

def gain_total_batch(Time1, Time2, GT, args):

    _, _, Dim = Time1.shape
    r = args.patches // 2
    Time1_mirror = np.pad(Time1, ((r,r), (r,r), (0,0)), mode='reflect')
    Time2_mirror = np.pad(Time2, ((r,r), (r,r), (0,0)), mode='reflect')

    Position_0 = np.array(np.argwhere(GT == 0))
    Position_1 = np.array(np.argwhere(GT == 1))
    Total_position = np.concatenate((Position_0, Position_1))
    print('    Total samples: ', Total_position.shape[0])

    ## Total samples
    total_T1 = np.zeros((Total_position.shape[0], args.patches, args.patches, Dim))
    total_T2 = np.zeros((Total_position.shape[0], args.patches, args.patches, Dim))
    total_gt = np.zeros((Total_position.shape[0],))
    for i in range(Total_position.shape[0]):
        temp_x, temp_y = Total_position[i,0], Total_position[i,1]
        total_T1[i,:,:,:] = gain_patch(Time1_mirror, temp_x, temp_y, patch_sizes=args.patches)
        total_T2[i,:,:,:] = gain_patch(Time2_mirror, temp_x, temp_y, patch_sizes=args.patches)
        total_gt[i]       = GT[temp_x, temp_y]

    Z_total  = torch.from_numpy(Total_position).type(torch.LongTensor)
    T1_total = torch.from_numpy(total_T1.transpose(0,3,1,2)).type(torch.FloatTensor)
    T2_total = torch.from_numpy(total_T2.transpose(0,3,1,2)).type(torch.FloatTensor)
    Y_total  = torch.from_numpy(total_gt).type(torch.LongTensor)
        
    batch_total = Data.TensorDataset(Z_total, T1_total, T2_total, Y_total)
    Total_loader = Data.DataLoader(dataset=batch_total,
                                    pin_memory=True,
                                    worker_init_fn=lambda worker_id: seed_worker(args.seed),
                                    batch_size=args.batches,
                                    shuffle=True,
                                    num_workers=6,)
    
    return Total_loader
    