import sys
import subprocess

#eval_file = sys.argv[1]

#print("File that will be evaluated", eval_file)

for eval_file in [
    "nti_fine_attenscaleprod_2comp_effembcapglovesmoothl1_noaug.txt",
    "nti_fine_attenscaleprod_3comp_effembcapglovesmoothl1_noaug.txt",
    "nti_fine_attenscaleprod_3compstaticw_effembcapglovesmoothl1_noaug.txt",
    "nti_fine_attenscaleprod_1comp_effembcapglovesmoothl1_noaug.txt"
]:

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=greedy_smoothl1_mmr", "--min_len=3"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=greedy_smoothl1_mmr", "--min_len=3"])


    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=3"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=3"])

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=2"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=2"])

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=1"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=1"])

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=3"])

    # # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    # #                 "--decodying_type=greedy_smoothl1_mmr", "--min_len=3",])

    # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=10", "--min_len=0", "--rep_window=0"])

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                         "--decodying_type=beam_wt_refinement", "--n_beam=10", "--min_len=0", "--rep_window=0",
    #                         "--eval_dataset_type=val"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=10", "--min_len=0", "--rep_window=0",
    #                 "--eval_dataset_type=val"])

    # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=10", "--min_len=0", "--rep_window=0",
    #                 "--eval_dataset_type=val"])


    # print("File that will be evaluated", eval_file)

    # # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
    # # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
    # # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

    # for beam_type in ["beam_comp", "beam_tutorial", "beam", "beam_wt_refinement"]:

        # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=10", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=10", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=10", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=1", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=1", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
        #                 "--decodying_type=" + beam_type, "--n_beam=1", "--min_len=0", "--rep_window=0",
        #                 "--eval_dataset_type=val"])

        # subprocess.run(["python3", "src/test_generate_sentences.py",
        #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
        # subprocess.run(["python2", "src/test_scores_coco.py",
        #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
        # subprocess.run(["python3", "src/test_scores_bertscore.py",
        #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])

    # for n_beam in [1, 2, 3, 5, 10]:
    #     print("n_beam", n_beam)

    #     n_beam = str(n_beam)

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam])
    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam])
    # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam])

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    # subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])

    # subprocess.run(["python3", "src/test_generate_sentences.py",
    #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
    # subprocess.run(["python2", "src/test_scores_coco.py",
    #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
    # subprocess.run(["python3", "src/test_scores_bertscore.py",
    #                 "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])

    # for rep_window in [1, 2, 3, 5]:
    #     rep_window = str(rep_window)
    #     for max_len in [10, 15, 20]:
    #         max_len = str(max_len)
    #         subprocess.run(
    #             ["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #              "--decodying_type=beam", "--n_beam=" + n_beam, "--max_len=" + max_len, "--rep_window=" + rep_window,
    #              "--eval_dataset_type=val"])
    #         subprocess.run(
    #             ["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #              "--decodying_type=beam", "--n_beam=" + n_beam, "--max_len=" + max_len, "--rep_window=" + rep_window,
    #              "--eval_dataset_type=val"])
    #         subprocess.run(
    #             ["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #              "--decodying_type=beam", "--n_beam=" + n_beam, "--max_len=" + max_len, "--rep_window=" + rep_window,
    #              "--eval_dataset_type=val"])
# greedy

sys.exit()
