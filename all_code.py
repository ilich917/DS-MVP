from __future__ import print_function
from __future__ import division

import time
import copy
import os
import urllib
import gc
import datetime
import warnings
import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import torch.onnx
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F

import cv2


from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import matplotlib.pyplot as plt
from PIL import Image

window = 20
dim_model = 128

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=window):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
    
class Data(Dataset):
    def __init__(self, df: pd.DataFrame, path: str, train: bool = True, preprocesar = None, context = None):
        self.preprocesar = preprocesar
        self.df = df
        self.path = path
        self.train = train
        self.context = context
        
    def __getitem__(self, index):
        im_path = os.path.join(self.path, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        meta = np.array(self.df.iloc[index][self.context].values, dtype=np.float32)

        if self.preprocesar:
            x = self.preprocesar(x)
            
        if self.train:
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
    
    def __len__(self):
        return len(self.df)
    
class Neural_Network(nn.Module):
    def __init__(self, pretrained, N_vocabulary: int):
        super(Neural_Network, self).__init__()

        self.body = pretrained
        self.face = pretrained
        self.handl = pretrained
        self.handr = pretrained


        #if 'ResNet' in str(pretrained.__class__):
        #self.pretrained.fc = nn.Linear(in_features=512, out_features=1, bias=True)

        if 'Inception3' in str(pretrained.__class__):
            self.body.AuxLogits.fc = nn.Linear(in_features = 768, out_features = 1)
            self.body.fc = nn.Linear(in_features = 2048, out_features = 512, bias = True)

            self.face.AuxLogits.fc = nn.Linear(in_features = 768, out_features = 1)
            self.face.fc = nn.Linear(in_features = 2048, out_features = 512, bias = True)
       
            self.handl.AuxLogits.fc = nn.Linear(in_features = 768, out_features = 1)
            self.handl.fc = nn.Linear(in_features = 2048, out_features = 512, bias = True)
       
            self.handr.AuxLogits.fc = nn.Linear(in_features = 768, out_features = 1)
            self.handr.fc = nn.Linear(in_features = 2048, out_features = 512, bias = True)
        
        self.body_plus_kp = nn.Sequential(nn.Linear(512 + n_body_kp, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
            
        self.face_plus_kp = nn.Sequential(nn.Linear(512 + n_face_kp, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        
        self.handl_plus_kp = nn.Sequential(nn.Linear(512 + n_handl_kp, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        
        self.handr_plus_kp = nn.Sequential(nn.Linear(512 + n_handr_kp, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        
        self.encoder = nn.Sequential(nn.Linear(1024, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.25),
                                  nn.Linear(512, 128),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))

        self.output = nn.Sequential(
            TransformerModel(window, 128, 8, 128, 4, 0.1),
                                  nn.Linear(128, N_vocabulary))

                                  
    def forward(self, inputs):
        """
        The input is a 20 frames video plus kp for each frame. 
        This is good, because when we will test this in longer or real time video,
        then we just need to feed it with chunks of 20 frames to make predictions.
        When frames in video are <20, then padding (when video is starting, for example) # NOT IMPLEMENTED YET
        """
              
        n = window # num of frames by video / to be defined

        #tensor to save the output of encoder for each frame
        distilled_video = torch.zeros(n, 128) # (n_frames, output_encoder)

        #videos of same length:
        body, face, handl, handr = inputs[0]

        # Following is the list of kp for each frame for each video [[kp_frame_1],[kp_frame_2],...,[kp_frame_n]]
        # where kp_frame_i is the list of kp coordinates normalized
        body_kp, face_kp, handl_kp, handr_kp = inputs[1]
        i = 0
        while i < n-1:
            #CNN forward
            body_cnn = self.body(body)
            face_cnn = self.face(face)
            handl_cnn = self.handl(handl)
            handr_cnn = self.handr(handr)

            #forward for CNN_output + kp
            body_pose = self.body_plus_kp(torch.cat((body_cnn, body_kp), dim=1))
            face_pose = self.face_plus_kp(torch.cat((face_cnn, face_kp), dim=1))
            handl_pose = self.handl_plus_kp(torch.cat((handl_cnn, handl_kp), dim=1))
            handr_pose = self.handr_plus_kp(torch.cat((handr_cnn, handr_kp), dim=1))

            #forward for encoding the concatenation of latest forwards
            distilled_video[i] = self.encoder(torch.cat(body_pose, face_pose, handl_pose, handr_pose))
            i += 1
        
        output = F.softmax(self.output(distilled_video))
        return output
    
    
pretrained = models.inception_v3(init_weights=False)
pretrained_weights = torch.load('../input/torchvision-inception-v3-imagenet-pretrained/inception_v3_google-1a9a5a14.pth', map_location='cpu')
pretrained.load_state_dict(pretrained_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 20
paciencia = 3

oof = np.zeros((len(train), 1))  # Out Of Fold predictions
preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)  # Predictions for test test

kf = KFold(n_splits=4, shuffle=True, random_state=47)

body, face, handl, handr, label = data #inception input dimensions of 299x299 
body_kp, face_kp, handl_kp, handr_kp = keypoints #csv #each cell is a vector (4 features) of keypoints all/face/handl/handr normalized

test_data = Data(df=test,
                       path = '../input/jpeg-melanoma-256x256/test', 
                       train=False,
                       preprocesar=preprocesar,
                       context=context)

for fold, (id_train, id_val) in enumerate(kf.split(X=np.zeros(len(train)), y=train['target']), 1):
    print('=' * 20, 'Fold', fold, '=' * 20) 
    model_path = f'model_{fold}.pth'  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    model = Red_Neuronal(pretrained=pretrained, n_meta_features=len(context))
    model = model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
    criterion = nn.BCEWithLogitsLoss()
    
    train_data = Data(df=train.iloc[id_train].reset_index(drop=True), 
                            path='../input/jpeg-melanoma-256x256/train/', 
                            train=True, 
                            preprocesar=preprocesar,
                            context=context)
    val = Data(df=train.iloc[id_val].reset_index(drop=True), 
                            path='../input/jpeg-melanoma-256x256/train/', 
                            train=True, 
                            preprocesar=preprocesar,
                            context=context)
    
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=2)
    
        
    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()
        
        for x, y in train_loader:
            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            optim.zero_grad()
            z, aux = model(x)
            
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(z, y.unsqueeze(1))
            loss2 = criterion(aux, y.unsqueeze(1))
            loss = loss1 + 0.4*loss2
            loss.backward()
            optim.step()
            pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
            correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
            epoch_loss += loss.item()
        train_acc = correct / len(id_train)
        
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(id_val), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
            val_acc = accuracy_score(train.iloc[id_val]['target'].values, torch.round(val_preds.cpu()))
            val_roc = roc_auc_score(train.iloc[id_val]['target'].values, val_preds.cpu())
            
            print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
            epoch + 1, 
            epoch_loss, 
            train_acc, 
            val_acc, 
            val_roc, 
            str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
            
            scheduler.step(val_roc)
                
            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                    break
                
    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(id_val), 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in enumerate(val_loader):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
        oof[id_val] = val_preds.cpu().numpy()
        
        # Predicting on test set
        tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
        for _ in range(TTA):
            for i, x_test in enumerate(test_loader):
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                tta_preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
        preds += tta_preds / TTA
    
preds /= kf.n_split