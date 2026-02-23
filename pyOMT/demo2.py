import os
import fnmatch
import csv
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import numpy as np
import scipy.io as sio
import argparse
import pdb

from networks import *
import P_loader

from pyOMT_raw import pyOMT_raw, train_omt


def gen_P(p_s, numX, output_P_gen, thresh=-1, topk=5, dissim=0.75, max_gen_samples=None):
    I_all = -torch.ones([topk, numX], dtype=torch.long)
    num_bat_x = numX//p_s.bat_size_n
    bat_size_x = min(numX, p_s.bat_size_n)
    for ii in range(max(num_bat_x, 1)):
        p_s.pre_cal(ii)
        p_s.cal_measure()
        _, I = torch.topk(p_s.d_U, topk, dim=0)
        for k in range(topk):
            I_all[k, ii*bat_size_x:(ii+1)*bat_size_x].copy_(I[k, 0:bat_size_x])
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long)
    for ii in range(topk-1):
        I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
        I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2
    
    
    
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    '''compute angles'''
    P = p_s.h_P      
    nm = torch.cat([P, -torch.ones(p_s.num_P,1)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]]), cs)
    theta = torch.acos(cs)
    # pdb.set_trace()


    '''filter out generated samples with theta larger than threshold'''
    I_gen = I_all[:, theta <= thresh]
    I_gen, _ = torch.sort(I_gen, dim=0)
    # _, uni_gen_id = np.unique(I_gen.numpy(), return_index=True, axis=1)
    _, uni_gen_id = np.unique(I_gen[0,:].numpy(), return_index=True)
    np.random.shuffle(uni_gen_id)
    I_gen = I_gen[:, torch.from_numpy(uni_gen_id)]
    # pdb.set_trace()
    
    numGen = I_gen.shape[1]
    if max_gen_samples is not None:
        numGen = min(numGen, max_gen_samples)
    I_gen = I_gen[:,:numGen]
    print('OT successfully generated {} samples'.format(
        numGen))
    
    '''generate new features'''
    # rand_w = torch.rand([numGen,1])    
    rand_w = dissim * torch.ones([numGen,1])
    P_gen = (torch.mul(P[I_gen[0,:],:], 1 - rand_w) + torch.mul(P[I_gen[1,:],:], rand_w)).numpy()

    P_gen2 = P[I_gen[0,:],:]
    P_gen = np.concatenate((P_gen,P_gen2))

    id_gen = I_gen[0,:].squeeze().numpy().astype(int)

    sio.savemat(output_P_gen, {'features':P_gen, 'ids':id_gen})

def compute_ot(
    input_P,
    output_h,
    output_P_gen,
    mode='train',
    thresh=0.7,
    topk=20,
    dissim=0.75,
    max_gen_samples=None,
    max_iter=20000,
    lr=5e-2,
    bat_size_n=1000,
    init_num_bat_n=20,
    num_gen_x=20000,
):
    '''args for omt'''
    TRAIN = False
    GENERATE = False
    if mode=='train':
        TRAIN = True
    elif mode=='generate':
        GENERATE = True
    else:
        print('unrecogonized OT computation action: ' + mode)

    h_P = torch.load(input_P)
    num_P = h_P.shape[0]
    dim_y = h_P.shape[1]
    maxIter = max_iter
    bat_size_P = num_P
    if not TRAIN:
        maxIter = 0


    #crop h_P to fit bat_size_P
    h_P = h_P[0:num_P//bat_size_P*bat_size_P,:]
    num_P = h_P.shape[0]

    p_s = pyOMT_raw(h_P, num_P, dim_y, maxIter, lr, bat_size_P, bat_size_n)
    '''train omt'''
    if TRAIN:
        train_omt(p_s, init_num_bat_n)
        torch.save(p_s.d_h, output_h)
    else:
        p_s.set_h(torch.load(output_h))

    if GENERATE:
        '''generate new samples'''
        gen_P(p_s, num_gen_x, output_P_gen, thresh=thresh, topk=topk, dissim=dissim, max_gen_samples=max_gen_samples)


if __name__ == "__main__":
    '''Arg parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ae", help="whether to train AE", dest='actions', action='append_const', const='train_ae')
    parser.add_argument("--refine_ae", help="whether to refine AE", dest='actions', action='append_const', const='refine_ae')
    parser.add_argument("--extract_feature", help="whether to extract latent code with AE encoder", dest='actions', action='append_const', const='extract_feature')
    parser.add_argument("--train_ot", help="whether to train (i.e. compute) OT with OT solver", dest='actions', action='append_const', const='train_ot')
    parser.add_argument("--generate_feature", help="whether to generate new latent codes", dest='actions', action='append_const', const='generate_feature')
    parser.add_argument("--decode_feature", help="whether to decode generated latent codes", dest='actions', action='append_const', const='decode_feature')

    parser.add_argument("--data_root_train", help='path to training set directory', type=str, metavar="", dest="data_root_train", default='/home/yang/Documents/data/CelebA_crop_resize_64/training')
    parser.add_argument("--data_root_test", help='path to testing set directory', type=str, metavar="", dest= "data_root_test",default='/home/yang/Documents/data/CelebA_crop_resize_64/testing')
    parser.add_argument("--ot_max_iter", help="max OT optimization iterations", type=int, default=20000)
    parser.add_argument("--ot_lr", help="OT solver learning rate", type=float, default=5e-2)
    parser.add_argument("--ot_bat_size_n", help="OT MC mini-batch size", type=int, default=1000)
    parser.add_argument("--ot_num_bat", help="OT MC mini-batch count per iteration", type=int, default=20)
    parser.add_argument("--ot_num_gen_x", help="number of OT latent samples to propose before filtering", type=int, default=20000)
    parser.add_argument("--seed", help="random seed for reproducibility", type=int, default=0)
    parser.add_argument("--result_root", help="root directory for training/generation outputs", type=str, default='./results')
    parser.add_argument("--num_workers", help="DataLoader workers", type=int, default=4)
    parser.add_argument("--image_size", help="final image size", type=int, default=64)
    parser.add_argument("--center_crop_size", help="center crop size before resize", type=int, default=178)
    parser.add_argument("--ae_num_epochs", help="AE training epochs", type=int, default=500)
    parser.add_argument("--ae_batch_size", help="AE batch size", type=int, default=512)
    parser.add_argument("--ae_learning_rate", help="AE learning rate", type=float, default=2e-5)
    parser.add_argument("--ae_dim_z", help="AE latent dimension", type=int, default=100)
    parser.add_argument("--ae_dim_f", help="AE base feature count", type=int, default=80)
    parser.add_argument("--ae_l1_lambda", help="AE latent L1 weight", type=float, default=1e-5)
    parser.add_argument("--ot_max_gen_samples", help="maximum OT samples after filtering", type=int, default=50000)
    parser.add_argument("--ot_angle_threshold", help="OT angle threshold", type=float, default=0.7)
    parser.add_argument("--ot_rec_gen_distance", help="OT interpolation dissimilarity", type=float, default=0.75)
    parser.add_argument("--decode_num_images", help="number of generated images to decode/save", type=int, default=50000)
    parser.add_argument("--device", help="compute device", type=str, default="cuda")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if args.actions is None:
        actions = ['train_ae', 'refine_ae', 'extract_feature', 'train_ot', 'generate_feature', 'decode_feature']
    else:
        actions = args.actions

    '''Training args'''
    RESUME = True #toggles of whether to resume training
    num_epochs = args.ae_num_epochs #max number of epochs for AE to train
    batch_size = args.ae_batch_size #batch size of AE training
    learning_rate = args.ae_learning_rate #learning rate of AE training
    dim_z = args.ae_dim_z #latent space dimension
    dim_c = 3 #input image number of channels
    dim_f = args.ae_dim_f #number of features in first layer of AE
    lmda = args.ae_l1_lambda #loss weight
    data_path = args.data_root_train #path to your training data folder (for AE)
    test_path = args.data_root_test #path to your testing data folder (for AE)

    '''Generation args'''
    max_gen_samples = args.ot_max_gen_samples #max number of generated samples. Used to avoid out of memory error.
    angle_threshold = args.ot_angle_threshold #angle threshold of OT generator ranging from [0,1]. See paper for details.
    rec_gen_distance = args.ot_rec_gen_distance #dis-similarity between reconstructed samples and generated samples, ranging from [0,1] with smaller meaning more similar
    
    '''Create directories'''
    result_root_path = args.result_root #root directory of training and generating results
    if not os.path.exists(result_root_path):
        os.mkdir(result_root_path)
    model_path = os.path.join(result_root_path, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    img_save_path = os.path.join(result_root_path, 'rec_imgs')
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    gen_im_path = os.path.join(result_root_path, 'gen_imgs')
    if not os.path.exists(gen_im_path):
        os.mkdir(gen_im_path) 
    gen_im_pair_path = os.path.join(result_root_path, 'gen_img_pairs')
    if not os.path.exists(gen_im_pair_path):
        os.mkdir(gen_im_pair_path)
    selected_model_path = os.path.join(result_root_path, 'saved_models')
    selected_ot_model_path = os.path.join(result_root_path, 'h.pt')
    feature_save_path = os.path.join(result_root_path, 'features.pt') 
    gen_feature_path = os.path.join(result_root_path, 'output_P_gen.mat')
        
    '''Start training and/or generating'''
    for action in actions:           
        img_transform = transforms.Compose([
            transforms.CenterCrop(args.center_crop_size),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        dataset = P_loader.P_loader(root=data_path,transform=img_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        testset = P_loader.P_loader(root=test_path,transform=img_transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
        model = autoencoder(dim_z, dim_c, dim_f).to(device)

        if action == 'train_ae':
            for test_data in testloader:
                test_img, _, _ = test_data
                break
            if RESUME:
                for file in os.listdir(model_path):
                    if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                        model.load_state_dict(torch.load(os.path.join(model_path, file), map_location=device))

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate)

            #save input test image
            save_image(test_img[:64], os.path.join(img_save_path, 'test_image_input.png'))
            for epoch in range(num_epochs):
                count_train = 0
                loss_train = 0.0
                count_test = 0
                loss_test = 0.0
                for data in dataloader:
                    img, _, _ = data
                    img = Variable(img).to(device)
                    # ===================forward=====================
                    output, z = model(img)
                    loss1 = criterion(output, img)
                    loss2 = torch.norm(z, 1)
                    loss = loss1 + lmda * loss2
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                    print('epoch [{}/{}], loss1:{:.4f}, loss2:{:.4f}'
                        .format(epoch, num_epochs, loss1.item(), loss2.item()))
                    loss_train += loss.item()
                    count_train += 1

                for data in testloader:
                    img, _, _ = data
                    img = Variable(img).to(device)
                    output, _ = model(img)
                    loss = criterion(output, img)
                    loss_test += loss.item()
                    count_test += 1

                loss_train /= count_train
                loss_test /= count_test
                out, _ = model(test_img.to(device))
                pic = out.data.cpu()
                save_image(pic[:64], os.path.join(img_save_path, 'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train, loss_test)))

                torch.save(model.state_dict(), os.path.join(model_path,'Epoch_{}_sim_autoencoder_{:04f}_{:04f}.pth'.format(epoch, loss_train, loss_test)))
        else:
            model_load_path = selected_model_path
            if (not os.path.exists(selected_model_path)) or len(os.listdir(selected_model_path)) == 0:
                model_load_path = model_path
            for file in os.listdir(model_load_path):
                if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                    model.load_state_dict(torch.load(os.path.join(model_load_path, file), map_location=device))

        if action == 'refine_ae':
            for param in model.block1.parameters():
                param.requires_grad = False
            for param in model.block2.parameters():
                param.requires_grad = False
            for param in model.block3.parameters():
                param.requires_grad = False
            for param in model.block4.parameters():
                param.requires_grad = False
            for param in model.block5.parameters():
                param.requires_grad = False

            for test_data in testloader:
                test_img, _, _ = test_data
                break
            if RESUME:
                for file in os.listdir(model_path):
                    if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                        model.load_state_dict(torch.load(os.path.join(model_path, file), map_location=device))

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate)

            #save input test image
            save_image(test_img[:64], os.path.join(img_save_path, 'test_image_input.png'))
            for epoch in range(num_epochs):
                count_train = 0
                loss_train = 0.0
                count_test = 0
                loss_test = 0.0
                for data in dataloader:
                    img, _, _ = data
                    img = Variable(img).to(device)
                    # ===================forward=====================
                    output, z = model(img)
                    loss = criterion(output, img)
                    # loss2 = torch.norm(z, 1)
                    # loss = loss1 + lmda * loss2
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================log========================
                    print('epoch [{}/{}], loss:{:.4f}'
                        .format(epoch, num_epochs, loss.item()))
                    loss_train += loss.item()
                    count_train += 1

                # for data in testloader:
                #     img, _, _ = data
                #     img = Variable(img).cuda()
                #     output, _ = model(img)
                #     loss = criterion(output, img)
                #     loss_test += loss.item()
                #     count_test += 1

                loss_train /= count_train
                # loss_test /= count_test
                loss_test = 0
                out, _ = model(test_img.to(device))
                pic = out.data.cpu()
                save_image(pic[:64], os.path.join(img_save_path, 'Epoch_{}_test_image_{:04f}_{:04f}.png'.format(epoch, loss_train, loss_test)))

                torch.save(model.state_dict(), os.path.join(model_path,'Epoch_{}_sim_refine_autoencoder_{:04f}_{:04f}.pth'.format(epoch, loss_train, loss_test)))
        else:
            model_load_path = selected_model_path
            if (not os.path.exists(selected_model_path)) or len(os.listdir(selected_model_path)) == 0:
                model_load_path = model_path
            for file in os.listdir(model_load_path):
                if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                    model.load_state_dict(torch.load(os.path.join(model_load_path, file), map_location=device))


        if action == 'extract_feature':
            dataloader_stable = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers,
            )
            features = torch.empty([len(dataset), dim_z], dtype=torch.float, requires_grad=False, device='cpu')
            i = 0
            for data in dataloader_stable:
                img, _, _ = data
                img = img.to(device)
                img.requires_grad = False
                # ===================forward=====================
                z = model.encoder(img.detach())
                features[i:i+img.shape[0], :] = z.squeeze().detach().cpu()
                i += img.shape[0]
                print('Extracted {}/{} features...'.format(i, len(dataset)))

            features = features[:i]
            torch.save(features, feature_save_path)


        if action == 'train_ot':
            compute_ot(
                feature_save_path,
                selected_ot_model_path,
                gen_feature_path,
                mode='train',
                max_gen_samples=max_gen_samples,
                thresh=angle_threshold,
                dissim=rec_gen_distance,
                max_iter=args.ot_max_iter,
                lr=args.ot_lr,
                bat_size_n=args.ot_bat_size_n,
                init_num_bat_n=args.ot_num_bat,
                num_gen_x=args.ot_num_gen_x,
            )
        elif action == 'generate_feature':
            ot_model_load_path = selected_ot_model_path
            if not os.path.exists(selected_ot_model_path):
                for file in os.listdir('.h/'):
                    if fnmatch.fnmatch(file, '*.pt'):
                        ot_model_load_path = os.path.join('.h/',file)
                        print('Successfully loaded OT model ' + ot_model_load_path)

            print('Generating features with OT solver...')
            compute_ot(
                feature_save_path,
                selected_ot_model_path,
                gen_feature_path,
                mode='generate',
                max_gen_samples=max_gen_samples,
                thresh=angle_threshold,
                dissim=rec_gen_distance,
                max_iter=args.ot_max_iter,
                lr=args.ot_lr,
                bat_size_n=args.ot_bat_size_n,
                init_num_bat_n=args.ot_num_bat,
                num_gen_x=args.ot_num_gen_x,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            
        if action == 'decode_feature':
            model_loaded = False
            for file in os.listdir(model_path):
                if fnmatch.fnmatch(file, 'Epoch_*_sim_refine_autoencoder*.pth'):
                    model.load_state_dict(torch.load(os.path.join(model_path, file), map_location=device))
                    model_loaded = True
            if not model_loaded:
                for file in os.listdir(model_path):
                    if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                        model.load_state_dict(torch.load(os.path.join(model_path, file), map_location=device))
                        model_loaded = True

            if not model_loaded:
                raise RuntimeError("No autoencoder checkpoint found for decode_feature.")

            feature_dict = sio.loadmat(gen_feature_path)
            features = feature_dict['features']
            ids = feature_dict['ids']

            if ids.ndim == 1:
                num_generated = int(ids.shape[0])
            else:
                num_generated = int(ids.shape[-1])

            decode_count = min(num_generated, int(args.decode_num_images))
            if decode_count <= 0:
                raise RuntimeError(
                    "decode_num_images must be positive and generated features must be available."
                )

            print(
                "Decoding generated features only: {} of {} available.".format(
                    decode_count, num_generated
                )
            )
            features = features[:decode_count, :]

            y_all = torch.empty([min(64, decode_count), 3, args.image_size, args.image_size])
            num_batches = (decode_count + batch_size - 1) // batch_size
            count = 0
            model.eval()
            for i in range(num_batches):
                batch_features = features[i * batch_size : (i + 1) * batch_size, :]
                if batch_features.shape[0] == 0:
                    continue
                z = torch.from_numpy(batch_features).to(device)
                z = z.view(batch_features.shape[0], -1, 1, 1)
                with torch.no_grad():
                    y = model.decoder(z)
                print('Decoding {}/{}...'.format(min((i + 1) * batch_size, decode_count), decode_count))
                for ii in range(y.shape[0]):
                    save_image(y[ii, :, :, :].cpu(), os.path.join(gen_im_path, 'gen_img_{0:06d}.png'.format(count)))
                    if count < y_all.shape[0]:
                        y_all[count] = y[ii, :, :, :].cpu()
                    count += 1

            print('Decoding complete. Saved {} images.'.format(count))
            save_image(y_all, os.path.join(gen_im_path, '..', 'gen_img.png'), nrow=8)
