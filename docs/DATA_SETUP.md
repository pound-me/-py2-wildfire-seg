# FLAME2 annotated subset

Use the preprocessed data and three-class masks released by RoboFireFuseNet:

https://drive.google.com/drive/folders/15bsStvQWBpMY1bXW3Wi-uliczz1-Zko8?usp=drive_link

Download the wildfire/FLAME2 content into:

```text
G:\py2\data\robofire
```

The expected package contains the split lists:

```text
G:\py2\data\robofire\lists\train_flm.txt
G:\py2\data\robofire\lists\val_flm.txt
G:\py2\data\robofire\lists\test_flm.txt
```

Each list entry is a path template containing `XXX`. The corresponding files
must replace it with:

- `rgb` for the UAV RGB image;
- `ir` for the infrared image;
- `gt` for the three-class segmentation mask.

For the first paper baseline, only the RGB image and ground-truth mask are used.
The infrared image is retained for later comparison but is not an input to the
proposed network.

After extraction, open `G:\py2` in VS Code and run the
`Check FLAME2 data` launch configuration.
