workdir = './model/model001'
seed = 20
apex = True

n_fold = 5
epoch = 3
resume_from = None

batch_size = 28
num_workers = 4
imgsize = (512, 512) #(height, width)

loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optim = dict(
    name='Adam',
    params=dict(
        lr=6e-5,
    ),
)

model = dict(
    name='se_resnext50_32x4d',
    pretrained='imagenet',
    n_output=6,
)

scheduler = dict(
    name='MultiStepLR',
    params=dict(
        milestones=[1,2],
        gamma=2/3,
    ),
)

#normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],}
normalize = None

crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize[0], width=imgsize[1]))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5,))
vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)

window_policy = 2

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds.pkl',
        imgdir='./input/stage_1_train_images',
        imgsize=imgsize,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy=window_policy,
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations='./cache/train_folds.pkl',
        imgdir='./input/stage_1_train_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy=window_policy,
    ),
    test = dict(
        dataset_type='CustomDataset',
        annotations='./cache/test.pkl',
        imgdir='./input/stage_1_test_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, contrast, totensor],
        dataset_policy='all',
        window_policy=window_policy,
    ),
)
