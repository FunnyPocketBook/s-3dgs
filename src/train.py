#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import cv2
from utils.clip_utils import CLIPEditor
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss 
from gaussian_renderer import calculate_selection_score, gsplat_render as render, network_gui
# from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, feature_map as feature_map_to_rgb, to_numpy
from encoding.utils import get_mask_pallete
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder
from models.semantic_dataloader import VariableSizeDataset
from torch.utils.data import DataLoader

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.object == "":
        print("Please specify the object name using --object.")
        exit()
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]

    
    # speed up for SAM
    if dataset.speedup:
        feature_in_dim = int(feature_out_dim/2)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        # MCMC only needs image, I only need feature_map and image. Rest is from the original for densification
        feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda().clone()
        gt_feature_map = viewpoint_cam.semantic_feature.cuda().clone()
        if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter: 
            # black_pixels = (image == 0)
            # gt_image[black_pixels] = 0
            mask = viewpoint_cam.mask.cuda()
            mask = mask.repeat(3, 1, 1)
            gt_image[~mask] = 0
            if iteration % 100 == 0:
                # viewpoint_cam = scene.getTrainCameras()[0]
                # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                # image_to_save = render_pkg["render"]
                # gt_image_to_save = viewpoint_cam.original_image.cuda()
                if not os.path.exists(f'{scene.model_path}/images/feature_map'):
                    os.makedirs(f'{scene.model_path}/images/feature_map')
                if not os.path.exists(f'{scene.model_path}/images/gt_feature_map'):
                    os.makedirs(f'{scene.model_path}/images/gt_feature_map')
                if not os.path.exists(f'{scene.model_path}/images/rgb'):
                    os.makedirs(f'{scene.model_path}/images/rgb')
                if not os.path.exists(f'{scene.model_path}/images/gt_rgb_mask'):
                    os.makedirs(f'{scene.model_path}/images/gt_rgb_mask')
                if not os.path.exists(f'{scene.model_path}/images/gt_rgb'):
                    os.makedirs(f'{scene.model_path}/images/gt_rgb')
                
                feature_map_rgb = feature_map_to_rgb(feature_map)
                feature_map_rgb_np = to_numpy(feature_map_rgb)
                cv2.imwrite(f'{scene.model_path}/images/feature_map/feature_map_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(feature_map_rgb_np, cv2.COLOR_BGR2RGB))
                
                gt_feature_map_rgb = feature_map_to_rgb(gt_feature_map)
                gt_feature_map_rgb_np = to_numpy(gt_feature_map_rgb)
                cv2.imwrite(f'{scene.model_path}/images/gt_feature_map/gt_feature_map_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(gt_feature_map_rgb_np, cv2.COLOR_BGR2RGB))
                
                image_np = to_numpy(image)
                cv2.imwrite(f'{scene.model_path}/images/rgb/image_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                
                gt_image_mask_np = to_numpy(gt_image)
                cv2.imwrite(f'{scene.model_path}/images/gt_rgb_mask/gt_image_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(gt_image_mask_np, cv2.COLOR_BGR2RGB))
                
                gt_image_np = (torch.clamp(viewpoint_cam.original_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                cv2.imwrite(f'{scene.model_path}/images/gt_rgb/gt_image_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(gt_image_np, cv2.COLOR_BGR2RGB))
        Ll1 = l1_loss(image, gt_image)
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
        if dataset.speedup:
            feature_map = cnn_decoder(feature_map)
        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature # Feature 3DGS
        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean() # MCMC
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean() # MCMC

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
                if dataset.speedup:
                    torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
  

            # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()
            

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                
                clip_editor = CLIPEditor()
                text_feature = clip_editor.encode_text([args.object])

                scores = calculate_selection_score(gaussians.get_semantic_feature[:, 0, :], text_feature, 
                                            score_threshold=0.5, positive_ids=[0])

                dead_mask = dead_mask | (scores < 1.0)

                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(args.cap_max, args.object)

            if iteration % 50 == 0:
                viewpoint_cam = scene.getTrainCameras()[0]
                rendered_image = render(viewpoint_cam, gaussians, pipe, background, override_image_scale=4)["render"]
                if not os.path.exists(f'{scene.model_path}/images/video_frames'):
                    os.makedirs(f'{scene.model_path}/images/video_frames')
                image_np = to_numpy(rendered_image)
                cv2.imwrite(f'{scene.model_path}/images/video_frames/image_{iteration}_{gaussians.get_opacity.shape[0]}.png', cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            
         

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.speedup:
                    cnn_decoder_optimizer.step()
                    cnn_decoder_optimizer.zero_grad(set_to_none = True)
                # MCMC ---------
                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)
                # ------- MCMC

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        
            if iteration % 100 == 0:
                viewpoint_cam = scene.getTrainCameras()[0]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_image_scale=6)
                image_to_save = render_pkg["render"]
                image_np = (torch.clamp(image_to_save, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                cv2.imwrite(f'{scene.model_path}/images/image_{iteration}.png', cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        # write net_image to file as an image using cv2
                        # cv2.imwrite('net_image.png', (torch.clamp(net_image, min=0, max=1.0) * 255).permute(1, 2, 0).cpu().numpy())
                        
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log,
                        # Add more metrics as needed
                        "iteration": iteration
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None
            


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2_000, 5_000, 7_000, 20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
