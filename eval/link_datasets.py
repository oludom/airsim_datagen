from glob import glob

# this file creates commands for creating symlinks from one folder to a folder of datasets
# so if ds1, ds2, ds3 each contain 2 tracks, so track0, track1, then you can create symlinks in ds called track0 to
# track5, linking to ds1/track0, ds1/track1, ds2/track0, ds2/track1, ds3/track0, ds3/track1 respectively

# fpöder to folders containing datasets (track0, track1, ...)
path_datasets = "/data/datasets/X1Gate8tracks"
# folder to link tracks to
path_linked = path_datasets + "/linked"
print(f"mkdir {path_linked}")

datasets = glob(path_datasets + "/*/")

track_count = 0

for dataset in datasets:
    # print(dataset)
    tracks = glob(dataset + "*/")

    for track in tracks:
        # print(track)
        # link track to linked dataset
        lp = path_linked + "/track" + str(track_count)
        track_count += 1
        print(f"ln -s {track} {lp}")