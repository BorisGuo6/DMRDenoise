import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree

import argparse
import math

from models.denoise import PointCloudDenoising
from models.utils import *

from tqdm import tqdm


def normalize_pointcloud(v):
    center = v.mean(axis=0, keepdims=True)
    v = v - center
    scale = (1 / np.abs(v).max()) * 0.999999
    v = v * scale
    return v, center, scale


def run_denoise(pc, patch_size, ckpt, device, random_state=0, expand_knn=16):
    pc, center, scale = normalize_pointcloud(pc)
    print('[INFO] Center: %s | Scale: %.6f' % (repr(center), scale))

    n_clusters = math.ceil(pc.shape[0] / patch_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(pc)

    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False, n_jobs=8)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    patches = []
    extra_points = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i
        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        extra_idx = np.setdiff1d(expand_idx, np.where(pts_idx))

        patches.append(pc[expand_idx])
        extra_points.append(pc[extra_idx])

    model = PointCloudDenoising.load_from_checkpoint(ckpt).to(device=device)

    denoised_patches = []
    downsampled_patches = []

    for patch in tqdm(patches):
        patch = torch.FloatTensor(patch).unsqueeze(0).to(device=device)
        # print(patch.size())
        with torch.no_grad():
            pred = model(patch)
            pred = pred.detach().cpu().reshape(-1, 3).numpy()

        denoised_patches.append(pred)

        downsampled_patches.append(model.model.adjusted.detach().cpu().reshape(-1, 3).numpy())

    denoised = np.concatenate(denoised_patches, axis=0)
    downsampled = np.concatenate(downsampled_patches, axis=0)

    denoised = (denoised / scale) + center
    downsampled = (downsampled / scale) + center

    return denoised, downsampled

def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []

    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter

        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)
    return clusters

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def run_denoise_middle_pointcloud(pc, num_splits, patch_size, ckpt, device, random_state=0, expand_knn=16):
    np.random.shuffle(pc)
    split_size = math.floor(pc.shape[0] / num_splits)
    splits = []
    for i in range(num_splits):
        if i < num_splits - 1:
            splits.append(pc[i*split_size:(i+1)*split_size])
        else:
            splits.append(pc[i*split_size:])

    denoised = []
    downsampled = []
    for i, splpc in enumerate(tqdm(splits)):
        den, dow = run_denoise(splpc, patch_size, ckpt, device, random_state, expand_knn)
        denoised.append(den)
        downsampled.append(dow)

    return np.vstack(denoised), np.vstack(downsampled)


def run_denoise_large_pointcloud(pc, cluster_size, patch_size, ckpt, device, random_state=0, expand_knn=16):
    n_clusters = math.ceil(pc.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=16).fit(pc)

    knn_graph = kneighbors_graph(pc, n_neighbors=expand_knn, mode='distance', include_self=False, n_jobs=8)
    knn_idx = np.array(knn_graph.tolil().rows.tolist())

    centers = []
    patches = []
    # extra_points = []
    for i in range(n_clusters):
        pts_idx = kmeans.labels_ == i

        raw_pc = pc[pts_idx]
        centers.append(raw_pc.mean(axis=0, keepdims=True))

        expand_idx = np.unique(knn_idx[pts_idx].flatten())
        # extra_idx = np.setdiff1d(expand_idx, np.where(pts_idx))

        patches.append(pc[expand_idx])
        # extra_points.append(pc[extra_idx])

        print('[INFO] Cluster Size:', patches[-1].shape[0])

    denoised = []
    downsampled = []
    for i, patch in enumerate(tqdm(patches)):
        den, dow = run_denoise(patch - centers[i], patch_size, ckpt, device, random_state, expand_knn)
        den += centers[i]
        dow += centers[i]
        denoised.append(den)
        downsampled.append(dow)

    return np.vstack(denoised), np.vstack(downsampled)


def run_test(input_fn, output_fn, patch_size, ckpt, device, random_state=0, expand_knn=16, ds_output_fn=None, large=False, cluster_size=30000):
    pc = np.loadtxt(input_fn).astype(np.float32)
    if not os.path.exists(os.path.dirname(output_fn)):
        os.makedirs(os.path.dirname(output_fn))
    if large:
        denoised, downsampled = run_denoise_large_pointcloud(pc, cluster_size, patch_size, ckpt, device, random_state=random_state, expand_knn=expand_knn)
    else:
        denoised, downsampled = run_denoise(pc, patch_size, ckpt, device, random_state=random_state, expand_knn=expand_knn)
    np.savetxt(output_fn, denoised)
    if ds_output_fn is not None:
        np.savetxt(ds_output_fn, downsampled)


def auto_denoise(args):
    print('[INFO] Loading: %s' % args.input)
    pc = np.load(args.input).astype(np.float32)
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    
    num_points = pc.shape[0]
    if num_points >= 120000:
        print('[INFO] Denoising large point cloud.')
        denoised, downsampled = run_denoise_large_pointcloud(
            pc=pc,
            cluster_size=args.cluster_size,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    elif num_points >= 60000:
        print('[INFO] Denoising middle-sized point cloud.')
        denoised, downsampled = run_denoise_middle_pointcloud(
            pc=pc,
            num_splits=args.num_splits,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    elif num_points >= 10000:
        print('[INFO] Denoising regular-sized point cloud.')
        denoised, downsampled = run_denoise(
            pc=pc,
            patch_size=args.patch_size,
            ckpt=args.ckpt,
            device=args.device,
            random_state=args.seed,
            expand_knn=args.expand_knn
        )
    else:
        assert False, "Our pretrained model does not support point clouds with less than 10K points."

    print(type(denoised))
    denoised_array = np.array(denoised)
    print(denoised_array.shape)

    denoised_array = regularize_pc_point_count(denoised_array, 400, use_farthest_point=args.freg)
    np.save(args.output, denoised_array)
    print('[INFO] Saving to: %s' % args.output)
    # if args.downsample_output is not None:
    #     np.savetxt(args.downsample_output, downsampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', type=str, default='./data/input_full_test_50k_0.010/airplane_0016.obj.xyz')
    parser.add_argument('--output', type=str, default='./airplane_0016.denoised.xyz')
    parser.add_argument('--ckpt', type=str, default='./pretrained/supervised/epoch=153.ckpt')
    parser.add_argument('--downsample_output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expand_knn', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--cluster_size', type=int, default=30000,
                        help='Number of clusters for large point clouds.')
    parser.add_argument('--num_splits', type=int, default=2,
                        help='Number of splits for middle-sized point clouds.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--freg', type=bool, default=True)
    args = parser.parse_args()
    
    auto_denoise(args)
