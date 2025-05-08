# Building3D Challenge

### Usage

Visualization script (requires vispy):

```
python visualize.py --dataset_path /path/to/dataset/
```

Example: `python visualize.py --dataset_path ./Building3D_entry_level/Entry-level/`

Simple training script:

```
python train.py --dataset /path/to/dataset/
```

Example: `python train.py --dataset ./Building3D_entry_level/Entry-level/`

### References

PC2WF: https://arxiv.org/pdf/2103.02766

Point2Roof: https://www.sciencedirect.com/science/article/pii/S0924271622002362

PBWR: https://arxiv.org/pdf/2311.12062

Building3D: http://building3d.ucalgary.ca/

### Notes

Wireframe file 17927 contained wrong lines ('# line 1') causing dataloader errors
Also other files --> fixed issue in dataloder `building3d.py`