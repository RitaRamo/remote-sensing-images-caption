import sys
import subprocess


#eval_file = sys.argv[1]

#print("File that will be evaluated", eval_file)

for eval_file in [
    "flickr_fine_encdec_2comp_effuflickr_noaug_notnormalized.txt",
    "ucml_fine_encdec_discrete_effucm_noaug_notnormalized.txt",
    "ucml_fine_encdec_1comp_effucm_noaug_notnormalized.txt",
    "ucml_fine_encdec_2comp_effucm_noaug_notnormalized.txt",
    "ucml_fine_encdec_3comp_effucm_noaug_notnormalized.txt",
    "ucml_fine_encdec_4comp_effucm_noaug_notnormalized.txt",
]:

    print("File that will be evaluated", eval_file)

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
    subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

    subprocess.run(["python3", "src/test_generate_sentences.py",
                    "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
    subprocess.run(["python2", "src/test_scores_coco.py",
                    "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])
    subprocess.run(["python3", "src/test_scores_bertscore.py",
                    "@experiments/conf_files/" + eval_file, "--eval_dataset_type=val"])

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
