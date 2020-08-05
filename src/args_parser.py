import argparse

from utils.enums import (
    EmbeddingsType,
    ImageNetModelsPretrained,
    OptimizerType,
    ContinuousLossesType,
    DecodingType,
    EvalDatasetType
)


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument(
        '--file_name', help='name of file that was used to fill the all other arguments', default="terminal")

    parser.add_argument(
        '--model_class_str', help='class name of the model to train', default="BasicEncoderDecoderModel")

    parser.add_argument('--pos_tag_dataset', action='store_true',
                        default=False, help='Use dataset of pos_tagging')

    parser.add_argument('--decodying_type', type=str, default=DecodingType.GREEDY.value,
                        choices=[decoding_type.value for decoding_type in DecodingType])

    parser.add_argument('--eval_dataset_type', type=str, default=EvalDatasetType.TEST.value,
                        choices=[dataset_type.value for dataset_type in EvalDatasetType],
                        help='dataset name for evaluating the model (compute metrics)',)

    parser.add_argument('--checkpoint_model', type=str, default=None)

    parser.add_argument('--augment_data', action='store_true',
                        default=False, help='Set a switch to true')

    # parser.add_argument('--test_set', action='store_true',
    #                     default=False, help='Set a switch to test set (otherwise validation set)')

    parser.add_argument('--post_processing', action='store_true',
                        default=False, help='Set a switch to true')

    parser.add_argument('--no_normalization', action='store_true',
                        default=False, help='Set a switch to false')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='define batch size to train the model')

    parser.add_argument('--n_beam', type=int, default=0,
                        help='define beam for inference')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='define num_works to dataloader')

    parser.add_argument('--fine_tune_encoder', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--set_cpu_device', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--optimizer_type', type=str, default=OptimizerType.ADAM.value,
                        choices=[optimizer.value for optimizer in OptimizerType])

    parser.add_argument('--encoder_lr', type=float, default=1e-4)

    parser.add_argument('--decoder_lr', type=float, default=4e-4)

    parser.add_argument('--epochs', type=int, default=40,
                        help='define epochs to train the model')

    parser.add_argument('--epochs_limit_without_improvement', type=int, default=12,
                        help='define the limit epoch of for early_stop')

    parser.add_argument(
        '--disable_steps', action='store_true', default=False,
        help='Conf just for testing: make the model run only 1 steps instead of the steps that was supposed')

    parser.add_argument('--image_model_type', type=str, default=ImageNetModelsPretrained.RESNET.value,
                        choices=[model.value for model in ImageNetModelsPretrained])

    parser.add_argument('--attention_dim', type=int,
                        default=512, help='define units of attention')

    parser.add_argument('--decoder_dim', type=int,
                        default=512, help='define units of decoder')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='define units of decoder')

    parser.add_argument('--disable_metrics', action='store_true', default=False,
                        help='Conf just for testing: make the model does not run the metrics')

    parser.add_argument('--continuous_loss_type', type=str, default=ContinuousLossesType.COSINE.value,
                        choices=[loss.value for loss in ContinuousLossesType])

    parser.add_argument('--embedding_type', type=str, default=None,
                        choices=[model.value for model in EmbeddingsType])

    parser.add_argument('--print_freq', type=int,
                        default=5, help='define print freq of loss')

    opts, _ = parser.parse_known_args()
    if opts.embedding_type == EmbeddingsType.GLOVE.value:
        parser.add_argument('--embed_dim', help='define dims of embeddings for words',
                            choices=(50, 100, 200, 300), default=50, type=int)
    elif opts.embedding_type == EmbeddingsType.FASTTEXT.value:
        parser.add_argument('--embed_dim', help='define dims of embeddings for words',
                            choices=(300,), default=300, type=int)
    elif opts.embedding_type == EmbeddingsType.CONCATENATE_GLOVE_FASTTEXT.value:
        parser.add_argument('--embed_dim', help='define dims of embeddings for words',
                            choices=(600,), default=600, type=int)
    elif opts.embedding_type == EmbeddingsType.BERT.value:
        parser.add_argument('--embed_dim', help='define dims of embeddings for words',
                            choices=(768,), default=768, type=int)
    else:
        parser.add_argument('--embed_dim', type=int, default=512,
                            help='define dims of embeddings for words')

    parser.add_argument('--w2', type=float, default=1.0,
                        help='define w of 4comp loss')

    parser.add_argument('--w3', type=float, default=1.0,
                        help='define w of 4comp loss')

    parser.add_argument('--w4', type=float, default=1.0,
                        help='define w of 4comp loss')

    args = parser.parse_args()

    return args
