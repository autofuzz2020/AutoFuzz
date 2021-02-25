import os
from ga_fuzzing import run_simulation
from object_types import pedestrian_types, vehicle_types, static_types, vehicle_colors
import random
import pickle
import numpy as np
from datetime import datetime
from customized_utils import make_hierarchical_dir, convert_x_to_customized_data, exit_handler, customized_routes, parse_route_and_scenario, check_bug, get_labels_to_encode, encode_fields, decode_fields, remove_fields_not_changing, recover_fields_not_changing, encode_bounds, max_one_hot_op, customized_standardize, customized_inverse_standardize, customized_fit, get_unique_bugs, get_if_bug_list, process_X, inverse_process_X, get_sorted_subfolders, load_data, get_picklename
import atexit

import traceback
from distutils.dir_util import copy_tree




import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models


from pgd_attack import train_net, pgd_attack, extract_embed

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--port', type=int, default=2045, help='TCP port(s) to listen to')
arguments = parser.parse_args()
port = arguments.port


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False


os.environ['HAS_DISPLAY'] = '0'
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 'lbc_augment', 'auto_pilot'
ego_car_model = 'lbc_augment'
is_save = True


# def get_gan_configs():
#
#     n_epoch = 200
#     batch_size = 64
#     lr = 0.0002
#     b1 = 0.5
#     b2 = 0.999
#     n_cpu = 8
#     latent_dim = 100
#     n_classes = 10
#     img_size = 32
#     channels = 1
#     sample_interval = 400
#
#
#     img_shape = (channels, img_size, img_size)
#
#     cuda = True if torch.cuda.is_available() else False
#
#
#     class Generator(nn.Module):
#         def __init__(self):
#             super(Generator, self).__init__()
#
#             self.label_emb = nn.Embedding(n_classes, n_classes)
#
#             def block(in_feat, out_feat, normalize=True):
#                 layers = [nn.Linear(in_feat, out_feat)]
#                 if normalize:
#                     layers.append(nn.BatchNorm1d(out_feat, 0.8))
#                 layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 return layers
#
#             self.model = nn.Sequential(
#                 *block(latent_dim + n_classes, 128, normalize=False),
#                 *block(128, 256),
#                 *block(256, 512),
#                 *block(512, 1024),
#                 nn.Linear(1024, int(np.prod(img_shape))),
#                 nn.Tanh()
#             )
#
#         def forward(self, noise, labels):
#             # Concatenate label embedding and image to produce input
#             gen_input = torch.cat((self.label_emb(labels), noise), -1)
#             img = self.model(gen_input)
#             img = img.view(img.size(0), *img_shape)
#             return img
#
#
#     class Discriminator(nn.Module):
#         def __init__(self):
#             super(Discriminator, self).__init__()
#
#             self.label_embedding = nn.Embedding(n_classes, n_classes)
#
#             self.model = nn.Sequential(
#                 nn.Linear(n_classes + int(np.prod(img_shape)), 512),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 512),
#                 nn.Dropout(0.4),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 512),
#                 nn.Dropout(0.4),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 1),
#             )
#
#         def forward(self, img, labels):
#             # Concatenate label embedding and image to produce input
#             d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
#             validity = self.model(d_in)
#             return validity
#
#
#     # Loss functions
#     adversarial_loss = torch.nn.MSELoss()
#
#     # Initialize generator and discriminator
#     generator = Generator()
#     discriminator = Discriminator()
#
#     if cuda:
#         generator.cuda()
#         discriminator.cuda()
#         adversarial_loss.cuda()
#
#     # Configure data loader
#     os.makedirs("../../data/mnist", exist_ok=True)
#     dataloader = torch.utils.data.DataLoader(
#         datasets.MNIST(
#             "../../data/mnist",
#             train=True,
#             download=True,
#             transform=transforms.Compose(
#                 [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#             ),
#         ),
#         batch_size=batch_size,
#         shuffle=True,
#     )
#
#     # Optimizers
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
#
#     FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
#
#
#     def sample_image(n_row, batches_done):
#         """Saves a grid of generated digits ranging from 0 to n_classes"""
#         # Sample noise
#         z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
#         # Get labels ranging from 0 to n_classes for n rows
#         labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#         labels = Variable(LongTensor(labels))
#         gen_imgs = generator(z, labels)
#         save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
#
#
#     # ----------
#     #  Training
#     # ----------
#     for epoch in range(n_epochs):
#         for i, (imgs, labels) in enumerate(dataloader):
#
#             batch_size = imgs.shape[0]
#
#             # Adversarial ground truths
#             valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#             fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
#
#             # Configure input
#             real_imgs = Variable(imgs.type(FloatTensor))
#             labels = Variable(labels.type(LongTensor))
#
#             # -----------------
#             #  Train Generator
#             # -----------------
#
#             optimizer_G.zero_grad()
#
#             # Sample noise and labels as generator input
#             z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
#             gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
#
#             # Generate a batch of images
#             gen_imgs = generator(z, gen_labels)
#
#             # Loss measures generator's ability to fool the discriminator
#             validity = discriminator(gen_imgs, gen_labels)
#             g_loss = adversarial_loss(validity, valid)
#
#             g_loss.backward()
#             optimizer_G.step()
#
#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#
#             optimizer_D.zero_grad()
#
#             # Loss for real images
#             validity_real = discriminator(real_imgs, labels)
#             d_real_loss = adversarial_loss(validity_real, valid)
#
#             # Loss for fake images
#             validity_fake = discriminator(gen_imgs.detach(), gen_labels)
#             d_fake_loss = adversarial_loss(validity_fake, fake)
#
#             # Total discriminator loss
#             d_loss = (d_real_loss + d_fake_loss) / 2
#
#             d_loss.backward()
#             optimizer_D.step()
#
#             print(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#             )
#
#             batches_done = epoch * len(dataloader) + i
#             if batches_done % sample_interval == 0:
#                 sample_image(n_row=10, batches_done=batches_done)










def rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model='lbc', x=[], record_every_n_step=10):
    is_bug = False

    # parameters preparation
    if ind == 0:
        launch_server = True
    else:
        launch_server = False

    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)['info']

        if len(x) == 0:
            x = d['x']
            # TBD: save port separately so we won't need to repetitvely save cur_info in ga_fuzzing
            x[-1] = port

        waypoints_num_limit = d['waypoints_num_limit']
        max_num_of_static = d['num_of_static_max']
        max_num_of_pedestrians = d['num_of_pedestrians_max']
        max_num_of_vehicles = d['num_of_vehicles_max']
        customized_center_transforms = d['customized_center_transforms']

        if 'parameters_min_bounds' in d:
            parameters_min_bounds = d['parameters_min_bounds']
            parameters_max_bounds = d['parameters_max_bounds']
        else:
            parameters_min_bounds = None
            parameters_max_bounds = None


        episode_max_time = 60
        call_from_dt = d['call_from_dt']
        town_name = d['town_name']
        scenario = d['scenario']
        direction = d['direction']
        route_str = d['route_str']
        route_type = d['route_type']


    folder = '_'.join([route_type, scenario, ego_car_model, route_str])
    route_info = customized_routes[route_type]
    location_list = route_info['location_list']
    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)




    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)
    print('x', x)


    objectives, loc, object_type, route_completion, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, rerun=True, record_every_n_step=record_every_n_step)
    print('\n'*10, 'save_path', save_path, '\n'*10)



    is_bug = int(check_bug(objectives))

    # save data
    if is_save:
        rerun_bugs_folder = make_hierarchical_dir([rerun_save_folder, folder, 'rerun_bugs'])
        rerun_non_bugs_folder = make_hierarchical_dir([rerun_save_folder, folder, 'rerun_non_bugs'])
        print('rerun_bugs_folder',rerun_bugs_folder)
        print('sub_folder_name', sub_folder_name)
        if is_bug:

            print('\n'*3, 'rerun also causes a bug!!!', '\n'*3)

            try:
                # use this version to merge into the existing folder
                copy_tree(save_path, os.path.join(rerun_bugs_folder, sub_folder_name))
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()
        else:
            try:
                # use this version to merge into the existing folder
                copy_tree(save_path, os.path.join(rerun_non_bugs_folder, sub_folder_name))
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()

    return is_bug, objectives



def rerun_list_of_scenarios(parent_folder, rerun_save_folder, scenario_file, data, mode, ego_car_model, record_every_n_step=10):
    import re
    if data == 'bugs':
        parent_folder_with_type = parent_folder + '/bugs'
    elif data == 'non_bugs':
        parent_folder_with_type = parent_folder + '/non_bugs'

    subfolder_names = get_sorted_subfolders(parent_folder, data)
    print('len(subfolder_names)', len(subfolder_names))
    _, _, objectives_list_np, _, _ = load_data(subfolder_names)
    collision_inds = objectives_list_np[:, 0] > 0.1
    subfolder_names = (np.array(subfolder_names)[collision_inds]).tolist()

    print('len(subfolder_names)', len(subfolder_names))

    if len(subfolder_names) > 200:
        subfolder_names = subfolder_names[:200]

    assert len(subfolder_names) >= 2
    mid = int(len(subfolder_names)//2)

    random.seed(0)
    random.shuffle(subfolder_names)


    train_subfolder_names = subfolder_names[:mid]
    test_subfolder_names = subfolder_names[mid:]

    print('len(train_subfolder_names)', len(train_subfolder_names))

    if mode == 'train':
        chosen_subfolder_names = train_subfolder_names
    elif mode == 'test':
        chosen_subfolder_names = test_subfolder_names
    elif mode == 'all':
        chosen_subfolder_names = subfolder_names

    bug_num = 0
    objectives_avg = 0

    for ind, sub_folder in enumerate(chosen_subfolder_names):
        print('episode:', ind+1, '/', len(chosen_subfolder_names), 'bug num:', bug_num)

        sub_folder_name = re.search(".*/([0-9]*)$", sub_folder).group(1)
        print('sub_folder', sub_folder)
        print('sub_folder_name', sub_folder_name)
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, 'cur_info.pickle')
            if os.path.exists(pickle_filename):
                print('pickle_filename', pickle_filename)
                is_bug, objectives = rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model=ego_car_model, record_every_n_step=record_every_n_step)


                objectives_avg += np.array(objectives)

                if is_bug:
                    bug_num += 1

    print('bug_ratio :', bug_num / len(chosen_subfolder_names))
    print('objectives_avg :', objectives_avg / len(chosen_subfolder_names))





if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


    scenario_folder = 'scenario_files'
    if not os.path.exists('scenario_files'):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    atexit.register(exit_handler, [port])

    # ['rerun', 'adv', 'tsne']
    task = 'rerun'
    use_adv = True
    use_unique_bugs = True
    test_unique_bugs = True
    unique_coeff = [0, 0.1, 0.5]

    parent_folder = 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/new_0.1_0.5_1000_500nsga2initial/2021_02_18_10_36_32,50_20_adv_nn_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.5__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01'

    pickle_filename = get_picklename(parent_folder)



    parent_tsne_folder = 'tmp_tsne'

    # only used for adv
    rerun_save_folder = make_hierarchical_dir(['adv', time_str])
    cutoff = 300
    cutoff_end = 350
    eps = 0.0
    adv_conf_th = 0.75
    # 1.01

    save_path = 'tmp_X_and_objectives'+'_'.join([str(use_unique_bugs), str(eps), str(adv_conf_th), str(unique_coeff[0]), str(unique_coeff[1]), str(unique_coeff[2]), str(cutoff), str(cutoff_end)])
    save_path = os.path.join(parent_tsne_folder, save_path)



    # 'tmp_X_and_objectives'
    # 'tmp_X_and_objectives_prev_all_True'
    if not os.path.exists(parent_tsne_folder):
        os.mkdir(parent_tsne_folder)


    consider_unique_for_tsne = True

    if task == 'rerun':
        # ['bugs', 'non_bugs']
        data = 'bugs'
        # ['train', 'test', 'all']
        mode = 'test'
        rerun_save_folder = make_hierarchical_dir(['rerun', data, mode, time_str])
        record_every_n_step = 1

        rerun_list_of_scenarios(parent_folder, rerun_save_folder, scenario_file, data, mode, ego_car_model, record_every_n_step=record_every_n_step)

    elif task == 'adv':


        with open(pickle_filename, 'rb') as f_in:
            d = pickle.load(f_in)
        # hack: since we are only using the elements after the first five

        xl_ori = d['xl']
        xu_ori = d['xu']
        customized_constraints = d['customized_constraints']


        subfolders = get_sorted_subfolders(parent_folder)

        initial_X, y, initial_objectives_list, mask, labels = load_data(subfolders)

        # objectives[0] > 0.1 or objectives[-3] or objectives[-2] or objectives[-1]
        # new_y = []
        # for ob in initial_objectives_list:
        #     # if ob[0] > 0.1:
        #     if ob[-2] == 1:
        #         new_y.append(1)
        #     else:
        #         new_y.append(0)
        # y = np.array(new_y)


        X_final_test = np.array(initial_X[cutoff:cutoff_end])


        unique_bugs = get_unique_bugs(initial_X[:cutoff], initial_objectives_list[:cutoff], mask, xl_ori, xu_ori, unique_coeff)
        unique_bugs_len = len(unique_bugs)

        # check out the initial additional unique bugs
        get_unique_bugs(initial_X[:cutoff_end], initial_objectives_list[:cutoff_end], mask, xl_ori, xu_ori, unique_coeff)


        partial = True



        X_train, X_test, xl, xu, labels_used, standardize, one_hot_fields_len, param_for_recover_and_decode = process_X(initial_X, labels, xl_ori, xu_ori, cutoff, cutoff_end, partial, unique_bugs_len)

        (X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields, _, _, unique_bugs_len) = param_for_recover_and_decode

        y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
        print('np.sum(y_train), np.sum(y_test)', np.sum(y_train), np.sum(y_test))

        model = train_net(X_train, y_train, X_test, y_test, batch_train=200, batch_test=2)

        print('\n'*3)
        train_conf = model.predict_proba(X_train)[:, 1]
        print('train_conf', sorted(train_conf))
        test_conf = model.predict_proba(X_test)[:, 1]
        print('test_conf', sorted(test_conf))
        th_conf = sorted(train_conf, reverse=True)[np.sum(y_train)//4]
        print(th_conf)
        print(np.sum(y_train), np.sum(y_test), np.sum(test_conf>th_conf))
        print('\n'*3)

        adv_conf_th = th_conf
        attack_stop_conf = np.max([0.75, th_conf])
        # adv_conf_th = 0.0

        if use_adv:
            y_zeros = np.zeros(X_test.shape[0])


            if use_unique_bugs:
                initial_test_x_adv_list, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(model, X_test, y_zeros, xl, xu, encoded_fields, labels_used, customized_constraints, standardize, prev_X=unique_bugs, base_ind=0, unique_coeff=unique_coeff, mask=mask, param_for_recover_and_decode=param_for_recover_and_decode, check_prev_x_all=True, eps=eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf)
            else:
                initial_test_x_adv_list, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(model, X_test, y_zeros, xl, xu, encoded_fields, labels_used, customized_constraints, standardize, eps=eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf)


            print('\n'*2)
            print('y_test :', y_test, 'total bug num :', np.sum(y_test))
            print('new_bug_pred_prob_list :', new_bug_pred_prob_list)
            print('initial_bug_pred_prob_list :', initial_bug_pred_prob_list)
            print('initial_bug_pred_prob_list median :', np.median(initial_bug_pred_prob_list))
            print(np.array(initial_test_x_adv_list).shape)



            X_final_test = inverse_process_X(np.array(initial_test_x_adv_list), standardize, one_hot_fields_len, partial, X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields)




        is_bug_list = []
        new_objectives_list = []

        for i, test_x_adv in enumerate(X_final_test):
            np.clip(test_x_adv, xl_ori, xu_ori)
            test_x_adv = np.append(test_x_adv, port)

            # print(test_x_adv)

            is_bug, objectives = rerun_simulation(pickle_filename, True, rerun_save_folder, i, str(i), scenario_file, ego_car_model=ego_car_model, x=test_x_adv)

            is_bug_list.append(is_bug)
            print(np.sum(is_bug_list), '/', len(is_bug_list))

            new_objectives_list.append(objectives)



        if use_adv:
            embed = extract_embed(model, np.concatenate([X_train, X_test, np.array(initial_test_x_adv_list)]))



            X_and_objectives = {
            'embed':embed,
            'initial_objectives_list':initial_objectives_list.tolist(),
            'cutoff':cutoff,
            'cutoff_end':cutoff_end,
            'new_objectives_list':new_objectives_list}
            np.savez(save_path, **X_and_objectives)


        print('is_bug_list :', is_bug_list)

        if test_unique_bugs:
            get_unique_bugs(initial_X[:cutoff]+X_final_test.tolist(), initial_objectives_list[:cutoff].tolist()+new_objectives_list, mask, xl_ori, xu_ori, unique_coeff)

        print('adv_conf_th', adv_conf_th)
        print('attack_stop_conf', attack_stop_conf)
        print(parent_folder)

    elif task == 'tsne':

        if consider_unique_for_tsne:
            with open(pickle_filename, 'rb') as f_in:
                d_info = pickle.load(f_in)
            # hack: since we are only using the elements after the first five

            xl_ori = d_info['xl']
            xu_ori = d_info['xu']
            subfolders = get_sorted_subfolders(parent_folder)
            initial_X, _, initial_objectives_list, mask, _ = load_data(subfolders)


        d = np.load(save_path+'.npz')
        cutoff = d['cutoff']
        cutoff_end = d['cutoff_end']
        embed = d['embed']



        prev_objectives_list = np.array(d['initial_objectives_list'][:cutoff])
        cur_objectives_list = np.array(d['initial_objectives_list'][cutoff:cutoff_end])
        cur_objectives_list_adv = np.array(d['new_objectives_list'])

        if_bug_list = get_if_bug_list(prev_objectives_list)
        if_bug_list_cur = get_if_bug_list(cur_objectives_list)
        if_bug_list_cur_adv = get_if_bug_list(cur_objectives_list_adv)





        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt
        X_embed = TSNE(n_components=2, perplexity=30.0, n_iter=3000).fit_transform(embed)



        prev_X = X_embed[:cutoff]

        if consider_unique_for_tsne:
            _, unique_bugs_inds = get_unique_bugs(initial_X[:cutoff], initial_objectives_list[:cutoff], mask, xl_ori, xu_ori, unique_coeff, return_indices=True)
            prev_X_bug_unique = prev_X[unique_bugs_inds]

        prev_X_normal = prev_X[if_bug_list==0]
        prev_X_bug = prev_X[if_bug_list==1]

        cur_X = X_embed[cutoff:cutoff_end]
        cur_X_normal = cur_X[if_bug_list_cur==0]
        cur_X_bug = cur_X[if_bug_list_cur==1]

        cur_X_adv = X_embed[cutoff_end:]
        cur_X_adv_normal = cur_X_adv[if_bug_list_cur_adv==0]
        cur_X_adv_bug = cur_X_adv[if_bug_list_cur_adv==1]


        print(prev_X.shape, cur_X.shape, cur_X_adv.shape)


        plt.scatter(prev_X_normal[:, 0], prev_X_normal[:, 1], c='yellow', label='prev X normal', alpha=0.5, s=2)
        plt.scatter(prev_X_bug[:, 0], prev_X_bug[:, 1], c='blue', label='prev X bug', alpha=0.5, s=2, marker='^')
        if consider_unique_for_tsne:
            plt.scatter(prev_X_bug_unique[:, 0], prev_X_bug_unique[:, 1], c='black', label='prev X unique bug', alpha=0.5, s=2, marker='^')

        plt.scatter(cur_X_normal[:, 0], cur_X_normal[:, 1], c='green', label='cur X normal', alpha=0.5, s=2)
        plt.scatter(cur_X_bug[:, 0], cur_X_bug[:, 1], c='green', label='cur X bug', alpha=0.5, s=2, marker='^')

        plt.scatter(cur_X_adv_normal[:, 0], cur_X_adv_normal[:, 1], c='red', label='cur X adv normal', alpha=0.5, s=2)
        plt.scatter(cur_X_adv_bug[:, 0], cur_X_adv_bug[:, 1], c='red', label='cur X adv bug', alpha=0.5, s=2, marker='^')


        for i in range(cur_X.shape[0]):
            plt.plot((cur_X[i, 0], cur_X_adv[i, 0]), (cur_X[i, 1], cur_X_adv[i, 1]), alpha=0.5, linewidth=0.5, c='gray')
        plt.legend(loc=2, prop={'size': 3}, framealpha=0.5)
        plt.savefig(os.path.join(parent_tsne_folder, 'tsne_'+save_path+'.pdf'))

# unique bugs num: 316 178 5 95 38
# unique bugs num: 339 193 5 100 41
# unique bugs num: 341 201 5 95 40
# unique bugs num: 347 206 5 95 41
