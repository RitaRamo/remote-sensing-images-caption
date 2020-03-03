import argparse
# from preprocess_data.images import ImageNetModelsPretrained
# from models.embeddings import EmbeddingsType
# from optimizers.optimizers import OptimizerType


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--augment_data', action='store_true',
                        default=False, help='Set a switch to true')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='define batch size to train the model')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='define num_works to dataloader')

    parser.add_argument('--fine_tune_encoder', action='store_true', default=False,
                        help='Set a switch to true')

    parser.add_argument('--encoder_lr', type=float, default=1e-4)

    parser.add_argument('--decoder_lr', type=float, default=4e-4)

    # parser.add_argument(
    #     '--file_name', help='name of file that was used to fill the all other arguments', default=None)

    # parser.add_argument(
    #     '--model_class_str', help='class name of the model to train', default="SimpleModel")

    # parser.add_argument('--image_model_type', type=str, default=ImageNetModelsPretrained.INCEPTION_V3.value,
    #                     choices=[model.value for model in ImageNetModelsPretrained])

    # parser.add_argument('--epochs', type=int, default=33,
    #                     help='define epochs to train the model')

    # parser.add_argument('--disable_steps', action='store_true', default=False,
    #                     help='Conf just for testing: make the model run only 1 steps instead of the steps that was supposed')

    # parser.add_argument('--disable_metrics', action='store_true', default=False,
    #                     help='Conf just for testing: make the model does not run the metrics')

    # parser.add_argument('--units', type=int, default=256,
    #                     help='define units to train the model')

    # parser.add_argument('--optimizer_type', type=str, default=OptimizerType.ADAM.value,
    #                     choices=[optimizer.value for optimizer in OptimizerType])

    # parser.add_argument('--optimizer_lr', type=float, default=0.001)

    # parser.add_argument('--embedding_type', type=str, default=None,
    #                     choices=[model.value for model in EmbeddingsType])

    # opts, _ = parser.parse_known_args()
    # if opts.embedding_type == EmbeddingsType.GLOVE.value:
    #     parser.add_argument('--embedding_size',
    #                         choices=(50, 100, 200, 300), default=50, type=int)
    # elif opts.embedding_type == EmbeddingsType.SPACY.value:
    #     parser.add_argument('--embedding_size', type=int, default=None)
    # else:
    #     parser.add_argument('--embedding_size', type=int, default=300)

    # parser.add_argument('--fine_tuning', action='store_true', default=False,
    #                     help='Set a switch to true')

    # parser.add_argument('--beam_search', action='store_true', default=False,
    #                     help='Set beam search or use default greedy')

    args = parser.parse_args()

    return args
