import sys
sys.path.append('src/')

from utils.utils import get_weights_with_parameter_sampler

if __name__ == "__main__":

    print("ola")

    param_list = get_weights_with_parameter_sampler()
    print("param list", param_list)

    for values in param_list:
        w2 = str(values["c2"])
        w3 = str(values["c3"])
        w4 = str(values["c4"])

        w2_percentage = "".join(w2.split(".") + ["0"])
        w3_percentage = "".join(w3.split(".") + ["0"])
        w4_percentage = "".join(w4.split(".") + ["0"])
        file_name = "split_fine_encdec_4comptvearly_eff_noaug_notnormalized1_" + w2_percentage + "_" + w3_percentage + "_" + w4_percentage + ".txt"
        with open('experiments/conf_files/' + file_name, 'w') as f:
            f.write('--file_name\n')
            f.write(file_name + "\n")
            f.write('--model_class_str\n')
            f.write('ContinuousEncoderDecoderImageWModel\n')
            f.write('--batch_size\n')
            f.write('8\n')
            f.write('--num_workers\n')
            f.write('0\n')
            f.write('--epochs\n')
            f.write('100\n')
            f.write('--dropout\n')
            f.write('0.5\n')
            f.write('--embedding_type\n')
            f.write('glove\n')
            f.write('--embed_dim\n')
            f.write('300\n')
            f.write('--continuous_loss_type\n')
            f.write('cos_avg_sentence_and_inputs_w_loss\n')
            f.write('--image_model_type\n')
            f.write('efficient_net\n')
            f.write('--no_normalization\n')
            f.write('--w2\n')
            f.write(w2 + '\n')
            f.write('--w3\n')
            f.write(w3 + '\n')
            f.write('--w4\n')
            f.write(w4 + '\n')
