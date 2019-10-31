from timeit import default_timer as dt

import numpy as np
import torch

from ..utils import layers
from ..model.loss import cosine_sim

from tqdm import tqdm


@torch.no_grad()
def predict_loader(model, data_loader, device,):

    model.eval()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None

    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x, total=len(x),
            desc='Pred  ',
            leave=False,
        )

    # for i, (images, captions, lengths, ids) in enumerate(data_loader):
    #     max_n_word = max(max_n_word, max(lengths))
    max_n_word = 77

    for batch in pbar_fn(data_loader):

        ids = batch['index']
        if len(batch['caption'][0]) == 2:
            (_, _), (_, lengths) = batch['caption']
        else:
            cap, lengths = batch['caption']
        img_emb, cap_emb = model.forward_batch(batch)

        if img_embs is None:
            if len(img_emb.shape) == 3:
                is_tensor = True
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            else:
                is_tensor = False
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy()
        if is_tensor:
            cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy()
        else:
            cap_embs[ids,] = cap_emb.data.cpu().numpy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]

    # Remove image feature redundancy
    if img_embs.shape[0] == cap_embs.shape[0]:
        img_embs = img_embs[
            np.arange(
                start=0,
                stop=img_embs.shape[0],
                step=data_loader.dataset.captions_per_image,
            ).astype(np.int),
        ]

    return img_embs, cap_embs, cap_lens


@torch.no_grad()
def evaluate(
    model, img_emb, txt_emb, lengths,
    device, shared_size=128, return_sims=False
):
    model.eval()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    begin_pred = dt()

    img_emb = torch.tensor(img_emb).to(device)
    txt_emb = torch.tensor(txt_emb).to(device)

    end_pred = dt()
    sims = model.compute_pairwise_similarity(
        model.similarity, img_emb, txt_emb, lengths,
        shared_size=shared_size
    )
    print(sims.min(), sims.max(), sims.mean())
        # sims = model.get_sim_matrix(
        #     embed_a=img_emb, embed_b=txt_emb,
        #     lens=lengths,
        # )
    div = sims.shape[1] / sims.shape[0]
    samp_sim = sims[:,np.arange(0, sims.shape[1], div).astype(np.int)]

    val_loss = model.multimodal_criterion(samp_sim)
    sims = layers.tensor_to_numpy(sims)

    end_sim = dt()

    i2t_metrics = i2t(sims)
    t2i_metrics = t2i(sims)

    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}

    metrics = {
        'pred_time': end_pred-begin_pred,
        'sim_time': end_sim-end_pred,
        'val_loss': val_loss,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum

    if return_sims:
        return metrics, sims

    return metrics


def i2t(sims,):
    """
    (images, captions)
    """

    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        begin = captions_per_image * index
        end = captions_per_image * index + captions_per_image
        for i in range(begin, end, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)


def t2i(sims,):
    """
    (images, captions)
    """

    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)
