import argparse
from embeddings.embeddings import EmbeddingsType


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument(
        '--file_name', help='name of file that was used to fill the all other arguments', default="terminal")

    parser.add_argument(
        '--model_class_str', help='class name of the model to train', default="BasicModel")

    parser.add_argument('--augment_data', action='store_true',
                        default=False, help='Set a switch to true')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='define batch size to train the model')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='define num_works to dataloader')

    parser.add_argument('--fine_tune_encoder', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--set_cpu_device', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--encoder_lr', type=float, default=1e-4)

    parser.add_argument('--decoder_lr', type=float, default=4e-4)

    parser.add_argument('--epochs', type=int, default=40,
                        help='define epochs to train the model')

    parser.add_argument('--epochs_limit_without_improvement', type=int, default=5,
                        help='define the limit epoch of for early_stop')

    parser.add_argument('--disable_steps', action='store_true', default=False,
                        help='Conf just for testing: make the model run only 1 steps instead of the steps that was supposed')

    parser.add_argument('--attention_dim', type=int,
                        default=512, help='define units of attention')

    parser.add_argument('--decoder_dim', type=int,
                        default=512, help='define units of decoder')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='define units of decoder')

    parser.add_argument('--disable_metrics', action='store_true', default=False,
                        help='Conf just for testing: make the model does not run the metrics')

    parser.add_argument('--embedding_type', type=str, default=None,
                        choices=[model.value for model in EmbeddingsType])

    parser.add_argument('--print_freq', type=int,
                        default=5, help='define print freq of loss')

    opts, _ = parser.parse_known_args()
    if opts.embedding_type == EmbeddingsType.GLOVE.value or opts.embedding_type == EmbeddingsType.GLOVE_FOR_CONTINUOUS_MODELS.value:
        parser.add_argument('--embed_dim', help='define dims of embeddings for words',
                            choices=(50, 100, 200, 300), default=50, type=int)
    else:
        parser.add_argument('--embed_dim', type=int, default=512,
                            help='define dims of embeddings for words')

    args = parser.parse_args()

    return args
