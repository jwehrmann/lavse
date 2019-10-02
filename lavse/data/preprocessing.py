from torchvision import transforms


def get_transform(
        cnn,
        split,
        resize_to=256,
        crop_size=224,
    ):

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    if cnn and 'inceptionv3' in cnn:
        resize_to = 299
        crop_size = 299
        means = [0.5, 0.5, 0.5]
        stds = [0.5, 0.5, 0.5]


    normalizer = transforms.Normalize(
        mean=means,
        std=stds,
    )

    if split == 'train':
        t_list = [
            # transforms.Resize(resize_to),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
        ]
    else:
        t_list = [
            transforms.Resize(resize_to),
            transforms.CenterCrop(crop_size),
        ]

    t_list.extend([transforms.ToTensor(), normalizer])
    transform = transforms.Compose(t_list)

    return transform

