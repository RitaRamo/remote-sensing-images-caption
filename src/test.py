import logging
from args_parser import get_args
from create_data_files import PATH_RSICD, PATH_DATASETS_RSICD, get_vocab_info, get_dataset
from models.basic_encoder_decoder_models.encoder_decoder import BasicEncoderDecoderModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.attention import BasicAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.sat import BasicShowAttendAndTellModel
from models.continuous_encoder_decoder_models.encoder_decoder import ContinuousEncoderDecoderModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention import ContinuousAttentionModel
from models.basic_encoder_decoder_models.encoder_decoder_variants.mask import BasicMaskGroundTruthWithPredictionModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_relu import ContinuousAttentionReluModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.bert import ContinuousBertModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling import ContinuousAttentionWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_sampling_alt import ContinuousAttentionWithScheduleSamplingAltModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image import ContinuousAttentionImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_schedule_alt_with_image import ContinuousAttentionImageWithScheduleSamplingModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_image_pos import ContinuousAttentionImagePOSModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image import ContinuousEncoderDecoderImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.enc_dec_image_and_classification import ContinuousEncoderDecoderImageClassificationModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_aoa_image import ContinuousAttentionAoAImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_sat_image import ContinuousSATImageModel
from models.continuous_encoder_decoder_models.encoder_decoder_variants.attention_aoanet_image import ContinuousAttentionAoANetImageModel
from torchvision import transforms
from PIL import Image
from preprocess_data.tokens import START_TOKEN, END_TOKEN
import numpy as np
import operator


import torch


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'


def generate_text(model, image, token_to_id, id_to_token, max_len):

    with torch.no_grad():  # no need to track history

        decoder_sentence = START_TOKEN + " "

        input_word = torch.tensor([token_to_id[START_TOKEN]])

        i = 1

        encoder_output = model.encoder(image)
        encoder_output = encoder_output.view(
            1, -1, encoder_output.size()[-1])

        h, c = model.decoder.init_hidden_state(encoder_output)

        while True:

            scores, h, c = model.generate_output_index(
                input_word, encoder_output, h, c)

            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)

            current_output_index = sorted_indices[0]

            current_output_token = id_to_token[current_output_index.item(
            )]

            decoder_sentence += " " + current_output_token

            if (current_output_token == END_TOKEN or
                    i >= max_len-1):  # until 35
                break

            input_word[0] = current_output_index.item()

            i += 1

        print("\ndecoded sentence", decoder_sentence)

        return decoder_sentence


def inference_with_beamsearch(model, image, token_to_id, id_to_token, max_len, n_solutions=3):

    def generate_n_solutions(seed_text, seed_prob, encoder_out,  h, c,  n_solutions):
        last_token = seed_text[-1]

        if last_token == END_TOKEN:
            return [(seed_text, seed_prob, h, c)]

        top_solutions = []
        scores, h, c = model.generate_output_index(
            torch.tensor([token_to_id[last_token]]), encoder_out, h, c)

        sorted_scores, sorted_indices = torch.sort(
            scores, descending=True, dim=-1)

        # 36,74

        # best_index_words = np.argsort(probs)[-n_solutions:][::-1]

        for index in range(n_solutions):
            text = seed_text + \
                [id_to_token[sorted_indices[index].item()]]
            # beam search taking into account lenght of sentence
            prob = (seed_prob*len(seed_text) + np.log(sorted_scores[index].item()) / (len(seed_text)+1))
            top_solutions.append((text, prob, h, c))

        return top_solutions

    def get_most_probable(candidates, n_solutions):
        return sorted(candidates, key=operator.itemgetter(1), reverse=True)[:n_solutions]

    with torch.no_grad():
        encoder_output = model.encoder(image)
        encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])  # flatten encoder
        h, c = model.decoder.init_hidden_state(encoder_output)

        top_solutions = [([START_TOKEN], 0.0, h, c)]

        for _ in range(max_len):
            candidates = []
            for sentence, prob, h, c in top_solutions:
                candidates.extend(generate_n_solutions(
                    sentence, prob, encoder_output, h, c,  n_solutions))

            top_solutions = get_most_probable(candidates, n_solutions)

        # print("top solutions", [(text, prob)
        #                         for text, prob, _, _ in top_solutions])

        best_tokens, prob, h, c = top_solutions[0]

        best_sentence = " ".join(best_tokens)

        print("\nbeam decoded sentence:", best_sentence)
        return best_sentence


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    device = torch.device("cpu")

    args = get_args()
    logging.info(args.__dict__)

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    test_dataset = get_dataset(PATH_DATASETS_RSICD+"test.json")

    model_class = globals()[args.model_class_str]
    model = model_class(
        args, vocab_size, token_to_id, id_to_token, max_len, device)
    model.setup_to_test()

    #scores = model.test(test_dataset)

    # # start test!
    predicted = {"args": [args.__dict__]}
    metrics = {}

    if args.disable_metrics:
        logging.info(
            "disable_metrics = True, thus will not compute metrics")
    else:
        nlgeval = NLGEval()  # loads the metrics models

    n_comparations = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                             std=[0.229, 0.224, 0.225])
    ])

    if args.beam_search:
        decoding_method = inference_with_beamsearch
        decoding_type = "beam"
    else:
        decoding_method = generate_text
        decoding_type = "greedy"

    for img_name, references in test_dataset.items():

        image_name = PATH_RSICD + \
            "raw_dataset/RSICD_images/" + img_name
        image = Image.open(image_name)
        image = transform(image)
        image = image.unsqueeze(0)

        model.decoder.eval()
        model.encoder.eval()

        text_generated = decoding_method(model, image, token_to_id, id_to_token, max_len)

        if args.disable_metrics:
            break

        # TODO:remove metrics that you will not use...
        all_scores = nlgeval.compute_individual_metrics(
            references, text_generated)

        if n_comparations % args.print_freq == 0:
            logging.info("this are dic metrics %s", all_scores)

        predicted[img_name] = {
            "value": text_generated,
            "scores": all_scores
        }

        for metric, score in all_scores.items():
            if metric not in metrics:
                metrics[metric] = score
            else:
                metrics[metric] += score
        n_comparations += 1

    avg_metrics = {metric: total_score /
                   n_comparations for metric, total_score in metrics.items()
                   }

    predicted['avg_metrics'] = {
        "value": "",
        "scores": avg_metrics
    }

    logging.info("avg_metrics %s", avg_metrics)

    model.save_scores(decoding_type, predicted)
