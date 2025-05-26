# visualize Building3D dataset

import os
import yaml
import torch
import argparse
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from Building3D.datasets import build_dataset

import vispy
import vispy.scene as scene

######################################################
DATASET_PATH = './Building3D_entry_level'
CONFIG_PATH = './Building3D/datasets/dataset_config.yaml'
SHARED_CAMERA = True
POINT_SIZE = 3
FOV = 45
BGCOLOR = 'black'
DISTANCE = 5
GT = True
#######################################################

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Building3D dataset')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Path to the dataset')
    parser.add_argument('--config_path', type=str, default=CONFIG_PATH, help='Path to the dataset config file')
    args = parser.parse_args()
    return args

def main(dataset_config):
    print('Building3D dataset config:', dataset_config)
    dataset_config['Building3D']['num_points'] = None # TODO: use original point cloud (to be handled in the dataset class)

    # setup visualization
    # Create a canvas and add a view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor=BGCOLOR)
    grid = canvas.central_widget.add_grid()

    # Create scatter plot
    view1 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view1, 0, 0)

    view2 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
    grid.add_widget(view2, 0, 1)

    markers1 = scene.visuals.Markers()
    markers2 = scene.visuals.Markers()

    # shared camera across all views
    shared_camera = scene.cameras.TurntableCamera(fov=FOV, azimuth=30, distance=DISTANCE)

    # Set view properties
    if SHARED_CAMERA:
        view1.camera = shared_camera
        view2.camera = shared_camera
    else:
        view1.camera = scene.cameras.TurntableCamera(fov=FOV, distance=DISTANCE)
        view2.camera = scene.cameras.TurntableCamera(fov=FOV, distance=DISTANCE)

    view1.add(markers1)
    view2.add(markers2)

    # Initialize the current file index
    index = 0

    # build dataset
    building3d_dataset = build_dataset(dataset_config.Building3D)

    # create dataloader
    train_loader = DataLoader(building3d_dataset['train'], batch_size=1, shuffle=False, drop_last=True, num_workers=4, collate_fn=building3d_dataset['train'].collate_batch)

    print('Dataset size: ', len(train_loader.dataset))

    # iterator
    iterator = iter(train_loader)

    edge_visual = None

    def update_view():
        nonlocal edge_visual
        #clear previous edges
        if edge_visual is not None and edge_visual.parent is not None:
            edge_visual.parent = None
        # get next batch
        batch = next(iterator)
        pc = batch['point_clouds'][0, :, :3].numpy()
        colors = batch['point_clouds'][0, :, 3:6].numpy()
        centroid = batch['centroid'][0].numpy() # (3, )
        scan_idx = batch['scan_idx']

        wf_vertices = batch['wf_vertices'][0].numpy()
        wf_edges = batch['wf_edges'][0].numpy().astype(np.int32)

        #print(f'Point cloud {scan_idx} shape: {pc.shape}')

        # check if vertices are in the point cloud
        # print(wf_vertices[0])
        # distances = np.linalg.norm(pc[:, :3] - wf_vertices[0], axis=1)
        # print(f'Min distance point: {pc[np.argmin(distances)]}')

        # print(f'Points: {len(pc)}')
        # print(wf_vertices.shape, wf_edges.shape)
        # print(wf_vertices)
        # print(wf_edges)

        edges = np.concatenate([[wf_vertices[i], wf_vertices[j]] for (i, j) in wf_edges], axis=0)

        assert len(wf_edges) == len(edges) // 2, f"wf_edges: {len(wf_edges)}, edges: {len(edges)}"

        # assert len(wf_edges) == len(edges), f"wf_edges: {len(wf_edges)}, edges: {len(edges)}"

        # plot also vertices in the point cloud view
        # pc = np.concatenate((pc, wf_vertices), axis=0)
        # colors = np.concatenate((colors, np.ones((len(wf_vertices), 3))), axis=0)

        if GT:
            class_labels = batch['class_label'][0].numpy()
            class_labels = class_labels.astype(bool)
            colors[class_labels] = [1, 0, 0]  # red

            print(f'Point cloud {scan_idx.item():<6} | size {len(pc):<6} | vertices {len(wf_vertices):<6} | edges {len(edges)//2:<6} | candidates {class_labels.sum():<6}')
        else:
            print(f'Point cloud {scan_idx.item():<6} | size {len(pc):<6} | vertices {len(wf_vertices):<6} | edges {len(edges)//2:<6}')


        markers1.set_data(pc, edge_color=colors, face_color=colors, size=POINT_SIZE)
        gt_color = 'black' if BGCOLOR == 'white' else 'white'
        markers2.set_data(wf_vertices, edge_color=gt_color, face_color=gt_color, size=POINT_SIZE+2)
        # add edges to the second view
        edge_visual = scene.visuals.Line(edges, connect='segments', color=gt_color, width=POINT_SIZE)
        view2.add(edge_visual)

    # Function to handle key press events
    @canvas.events.key_press.connect
    def on_key_press(event):
        # Check if the 'n' key is pressed
        if event.key == 'n':
            # Update the point cloud visualization
            update_view()
        elif event.key == 'q':
            # Close the canvas
            exit()

    update_view()  # Initial update
    # Start the event loop
    vispy.app.run()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    dataset_config = cfg_from_yaml_file(args.config_path)
    dataset_config['Building3D']['root_dir'] = args.dataset_path
    #dataset_config['Building3D']['num_points'] = None # use original point cloud

    main(dataset_config)