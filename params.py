import argparse
from addict import Dict
from lavse.model.similarity.factory import get_sim_names
from lavse.model import imgenc, txtenc
import profiles


def get_train_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'options',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--ngpu', type=int, default=1)
    # parser.add_argument(
    #     '--data_path',
    # )
    # parser.add_argument(
    #     '--train_data', default='f30k_precomp.en',
    #     help=(
    #         'Data used to align images and captions.'
    #         'Eg.: f30k_precomp.en'
    #     ),
    # )
    # parser.add_argument(
    #     '--val_data', default=['f30k_precomp.en'], nargs='+',
    #     help=(
    #         'Data used for evaluation during training.'
    #         'Eg.: [f30k_precomp.en,m30k_precomp.de]'
    #     ),
    # )
    # parser.add_argument(
    #     '--adapt_data', default=None, nargs='+',
    #     help=(
    #         'Data used for training joint language space.'
    #         'Eg.: [m30k_precomp.en-de,jap_precomp.en-jt]'
    #     ),
    # )
    # parser.add_argument(
    #     '--vocab_path', default='./vocab/complete.json',
    #     help='Path to saved vocabulary json files.',
    # )
    # parser.add_argument(
    #     '--margin', default=0.2, type=float,
    #     help='Rank loss margin.',
    # )
    # parser.add_argument(
    #     '--num_epochs', default=30, type=int,
    #     help='Number of training epochs.',
    # )
    # parser.add_argument(
    #     '--device', default='cuda:0', type=str,
    #     help='Device to run the model.',
    # )
    # parser.add_argument(
    #     '--sim', default='cosine', type=str,
    #     help='Similarity.', choices=get_sim_names(),
    # )
    # parser.add_argument(
    #     '--batch_size', default=128, type=int,
    #     help='Size of a training mini-batch.',
    # )
    # parser.add_argument(
    #     '--embed_dim', default=300, type=int,
    #     help='Dimensionality of the word embedding.',
    # )
    # parser.add_argument(
    #     '--latent_size', default=1024, type=int,
    #     help='Dimensionality of the joint embedding.',
    # )
    # parser.add_argument(
    #     '--grad_clip', default=2., type=float,
    #     help='Gradient clipping threshold.',
    # )
    # parser.add_argument(
    #     '--outpath',
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--resume', default=None,
    #     help='Checkpoint to resume training.',
    # )
    # parser.add_argument(
    #     '--data_parallel', action='store_true',
    #     help=(
    #         'Whether to use several GPUs to train the model',
    #         '[use when a single model do not fit the GPU memory].'
    #     ),
    # )
    # parser.add_argument(
    #     '--profile', default=None,
    #     choices=profiles.get_profile_names(),
    #     help='Import pre-defined setup from profiles.py',
    # )
    # parser.add_argument(
    #     '--text_encoder', default='gru',
    #     choices=txtenc.get_available_txtenc(),
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--text_pooling', default='lens',
    #     choices=['mean', 'max', 'lens', 'none'],
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--image_encoder', default='scan',
    #     choices=imgenc.get_available_imgenc(),
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--image_pooling', default='mean',
    #     choices=['mean', 'max', 'lens', 'none'],
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--text_repr',
    #     default='word',
    #     help='Path to save logs and models.',
    # )
    # parser.add_argument(
    #     '--lr', default=.0002, type=float,
    #     help='Initial learning rate.',
    # )
    # parser.add_argument(
    #     '--lr_decay_interval', default=15, type=int,
    #     help='Number of epochs to update the learning rate.',
    # )
    # parser.add_argument(
    #     '--lr_decay_rate', default=0.1, type=float,
    #     help='Number of epochs to update the learning rate.',
    # )
    # parser.add_argument(
    #     '--workers', default=1, type=int,
    #     help='Number of data loader workers.',
    # )
    # parser.add_argument(
    #     '--log_step', default=10, type=int,
    #     help='Number of steps to print and record the log.',
    # )
    # parser.add_argument(
    #     '--nb_epochs', default=45, type=int,
    #     help='Number of epochs.',
    # )
    # parser.add_argument(
    #     '--early_stop', default=30, type=int,
    #     help='Early stop patience.',
    # )
    # parser.add_argument(
    #     '--valid_interval', default=500, type=int,
    #     help='Number of steps to run validation.',
    # )
    # parser.add_argument(
    #     '--max_violation', action='store_true',
    #     help='Use max instead of sum in the rank loss (i.e., k=1)',
    # )
    # parser.add_argument(
    #     '--increase_k', default=.0, type=float,
    #     help='Rate for linear increase of k hyper-parameter (used when not --max_violation). ',
    # )
    # parser.add_argument(
    #     '--initial_k', default=1., type=float,
    #     help='Initial value for k hyper-parameter (used when not --max_violation)',
    # )
    # parser.add_argument(
    #     '--beta', default=0.995, type=float,
    #     help='Initial value for k hyper-parameter (used when not --max_violation)',
    # )
    # parser.add_argument(
    #     '--log_level', default='info',
    #     choices=['debug', 'info'],
    #     help='Log/verbosity level.',
    # )
    # parser.add_argument(
    #     '--eval_before_training', action='store_true',
    #     help='Performs complete eval before training',
    # )
    # parser.add_argument(
    #     '--save_all', action='store_true',
    #     help='Save checkpoints for all models',
    # )
    # parser.add_argument(
    #     '--finetune', action='store_true',
    #     help='Finetune convolutional net',
    # )

    # parser.add_argument(
    #     '--loader_name', default='precomp',
    #     help='Loader to be used',
    # )
    args = parser.parse_args()
    args = Dict(vars(args))
    return args
