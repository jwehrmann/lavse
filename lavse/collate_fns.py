import torch
import numpy as np


def default_padding(captions):
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return targets, lengths


def collate_fn_word(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    targets, lengths = default_padding(captions)
    
    return images, targets, lengths, ids


def collate_lang_word(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    lang_a, lang_b, ids = zip(*data)
    
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    targ_a, lens_a = default_padding(lang_a)
    targ_b, lens_b = default_padding(lang_b)

    return targ_a, lens_a, targ_b, lens_b, ids


def collate_fn_liwe(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    chars_pad, lengths = liwe_padding(captions)

    return images, chars_pad, lengths, ids


def collate_lang_liwe(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    lang_a, lang_b, ids = zip(*data)
    
    # lens_a = np.array([len(cap) for cap in words_a])
    # lens_b = np.array([len(cap) for cap in words_b])

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    targ_a, lens_a = liwe_padding(lang_a)
    targ_b, lens_b = liwe_padding(lang_b)

    return targ_a, lens_a, targ_b, lens_b, ids


def liwe_padding(captions):
    splitted_caps = []
    for caption in captions:    
        sc = split_array(caption) 
        splitted_caps.append(sc) 
    sent_lens = np.array([len(x) for x in splitted_caps])
    max_nb_steps = max(sent_lens)
    word_maxlen = 26
    targets = torch.zeros(len(captions), max_nb_steps, word_maxlen).long()    
    for i, cap in enumerate(splitted_caps):    
        end_sentence = sent_lens[i]    
        for j, word in enumerate(cap):
            end_word = word_maxlen if len(word) > word_maxlen else len(word)
            targets[i, j, :end_word] = word[:end_word]

    return targets, sent_lens


def split_array(iterable, splitters=[0, 1, 2]):
    import itertools            
    return [
        torch.LongTensor(list(g))
        for k, g in itertools.groupby(
            iterable, lambda x: x in splitters
        ) 
        if not k
    ]
