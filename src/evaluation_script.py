import sys
import subprocess

#eval_file = sys.argv[1]

#print("File that will be evaluated", eval_file)

#alpha_consistency=0.20
# Bleu_1: 0.775
# Bleu_2: 0.671
# Bleu_3: 0.585
# Bleu_4: 0.514
# computing METEOR score...
# METEOR: 0.365
# computing Rouge score...
# ROUGE_L: 0.676
# computing CIDEr score...
# CIDEr: 2.734

#0.15
# Bleu_1: 0.776
# Bleu_2: 0.672
# Bleu_3: 0.585
# Bleu_4: 0.514
# computing METEOR score...
# METEOR: 0.366
# computing Rouge score...
# ROUGE_L: 0.676
# computing CIDEr score...
# CIDEr: 2.735

# #0.10
# Bleu_1: 0.777
# Bleu_2: 0.673
# Bleu_3: 0.586
# Bleu_4: 0.515
# computing METEOR score...
# METEOR: 0.367
# computing Rouge score...
# ROUGE_L: 0.676
# computing CIDEr score...
# CIDEr: 2.735

# 0.05
# Bleu_1: 0.782
# Bleu_2: 0.678
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.370
# computing Rouge score...
# ROUGE_L: 0.681
# computing CIDEr score...
# CIDEr: 2.767

# 0.03
# Bleu_1: 0.783
# Bleu_2: 0.678
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.371
# computing Rouge score...
# ROUGE_L: 0.681
# computing CIDEr score...
# CIDEr: 2.768

# 0.02
# Bleu_1: 0.784
# Bleu_2: 0.679
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.371
# computing Rouge score...
# ROUGE_L: 0.681
# computing CIDEr score...
# CIDEr: 2.777

# 0.01
# Bleu_1: 0.784
# Bleu_2: 0.679
# Bleu_3: 0.591
# Bleu_4: 0.518
# computing METEOR score...
# METEOR: 0.371
# computing Rouge score...
# ROUGE_L: 0.680
# computing CIDEr score...
# CIDEr: 2.775

# #0.50
# Bleu_1: 0.764
# Bleu_2: 0.659
# Bleu_3: 0.572
# Bleu_4: 0.502
# computing METEOR score...
# METEOR: 0.358
# computing Rouge score...
# ROUGE_L: 0.666
# computing CIDEr score...
# CIDEr: 2.680

# #ver uc

# New beam
# 0.02
# Bleu_1: 0.785
# Bleu_2: 0.680
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.372
# computing Rouge score...
# ROUGE_L: 0.680
# computing CIDEr score...
# CIDEr: 2.778

# 0.03
# Bleu_1: 0.785
# Bleu_2: 0.679
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.371
# computing Rouge score...
# ROUGE_L: 0.681
# computing CIDEr score...
# CIDEr: 2.778

# 0.04
# leu_1: 0.785
# Bleu_2: 0.680
# Bleu_3: 0.592
# Bleu_4: 0.520
# computing METEOR score...
# METEOR: 0.372
# computing Rouge score...
# ROUGE_L: 0.682
# computing CIDEr score...
# CIDEr: 2.780


# 0.05
# Bleu_1: 0.784
# Bleu_2: 0.679
# Bleu_3: 0.592
# Bleu_4: 0.520
# computing METEOR score...
# METEOR: 0.372
# computing Rouge score...
# ROUGE_L: 0.681
# computing CIDEr score...
# CIDEr: 2.778


# 0.10
# Bleu_1: 0.783
# Bleu_2: 0.678
# Bleu_3: 0.591
# Bleu_4: 0.519
# computing METEOR score...
# METEOR: 0.371
# computing Rouge score...
# ROUGE_L: 0.680
# computing CIDEr score...
# CIDEr: 2.767


# 0.02
# Bleu_1: 0.829
# Bleu_2: 0.753
# Bleu_3: 0.693
# Bleu_4: 0.636
# computing METEOR score...
# METEOR: 0.407
# computing Rouge score...
# ROUGE_L: 0.764
# computing CIDEr score...
# CIDEr: 3.040

# 0.00
# Bleu_1: 0.848
# Bleu_2: 0.779
# Bleu_3: 0.722
# Bleu_4: 0.667
# computing METEOR score...
# METEOR: 0.432
# computing Rouge score...
# ROUGE_L: 0.788
# computing CIDEr score...
# CIDEr: 3.187

# 0.01
# Bleu_1: 0.841
# Bleu_2: 0.770
# Bleu_3: 0.710
# Bleu_4: 0.654
# computing METEOR score...
# METEOR: 0.421
# computing Rouge score...
# ROUGE_L: 0.780
# computing CIDEr score...
# CIDEr: 3.138

# 0.04
# Bleu_1: 0.830
# Bleu_2: 0.757
# Bleu_3: 0.694
# Bleu_4: 0.633
# computing METEOR score...
# METEOR: 0.408
# computing Rouge score...
# ROUGE_L: 0.761
# computing CIDEr score...
# CIDEr: 3.021



# #beam3, n_min4, alpha=0.1
# Bleu_1: 0.850
# Bleu_2: 0.780
# Bleu_3: 0.722
# Bleu_4: 0.666
# computing METEOR score...
# METEOR: 0.430
# computing Rouge score...
# ROUGE_L: 0.787
# computing CIDEr score...
# CIDEr: 3.199

# 0.02
# Bleu_1: 0.852
# Bleu_2: 0.782
# Bleu_3: 0.723
# Bleu_4: 0.667
# computing METEOR score...
# METEOR: 0.431
# computing Rouge score...
# ROUGE_L: 0.787
# computing CIDEr score...
# CIDEr: 3.199


# 0.03
# Bleu_1: 0.851
# Bleu_2: 0.781
# Bleu_3: 0.723
# Bleu_4: 0.667
# computing METEOR score...
# METEOR: 0.430
# computing Rouge score...
# ROUGE_L: 0.786
# computing CIDEr score...
# CIDEr: 3.201



# alpha 0.04
# Bleu_1: 0.848
# Bleu_2: 0.778
# Bleu_3: 0.721
# Bleu_4: 0.666
# computing METEOR score...
# METEOR: 0.429
# computing Rouge score...
# ROUGE_L: 0.786
# computing CIDEr score...
# CIDEr: 3.200


for eval_file in [
    #"nti_fine_attenscaleprod_2comp_effembcapglovesmoothl1_noaug.txt",
    #"nti_fine_attenscaleprod_3comp_effembcapglovesmoothl1_noaug.txt",
    
    "ucmti_fine_attenscaleprod_3compgnr1staticend_effembcapglovesmoothl1_noaug.txt",
    "nti_fine_attenscaleprod_3compstaticw_effembcapglovesmoothl1_noaug.txt"
]:

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=greedy_smoothl1_mmr"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=greedy_smoothl1_mmr"])

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=6", "--rep_window=2"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=6", "--rep_window=2"])




    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=4", "--rep_window=2"])

    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=4", "--rep_window=2"])


   


    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=2", "--rep_window=2"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=2", "--rep_window=2"])


    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=3", "--rep_window=2"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=3", "--rep_window=2"])
    
    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=4", "--rep_window=2"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=4", "--rep_window=2"])
    
    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=5", "--rep_window=2"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=5", "--min_len=5", "--rep_window=2"])
    
    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam_wt_refinement", "--n_beam=3"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=beam_wt_refinement", "--n_beam=3"])
    
    
    
    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=3"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=3"])

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=2"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=2"])

    # subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=1"])

    # subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                 "--decodying_type=greedy_smoothl1_mmr", "--min_len=3", "--rep_window=1"])

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
