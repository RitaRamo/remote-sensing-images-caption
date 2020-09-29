import sys
import subprocess


#eval_file = sys.argv[1]

#print("File that will be evaluated", eval_file)

for eval_file in [
    "flick_fine_encdec_discrete_effimgnt_noaug_notnormalized.txt",
    # "flick_fine_encdec_discrete_effflick_noaug_notnormalized.txt",
    "flickr_fine_encdec_1comp_effflickr_noaug_notnormalized.txt",
    # "flickr_fine_encdec_4comp_effuflickr_noaug_notnormalized.txt"
]:

    print("File that will be evaluated", eval_file)

    # for n_beam in [1, 3, 5, 15]:
    #     print("n_beam", n_beam)

    #     n_beam = str(n_beam)

    #     subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam])
    #     subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam])
    #     subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam])

    #     subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    #     subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    #     subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
    #                     "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])

    n_beam = "10"
    for min_len in [2, 3, 5, 7, 10]:
        min_len = str(min_len)
        subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam", "--n_beam=" + n_beam, "--min_len=" + min_len, "--eval_dataset_type=val"])
        subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam", "--n_beam=" + n_beam, "--min_len=" + min_len, "--eval_dataset_type=val"])
        subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam", "--n_beam=" + n_beam, "--min_len=" + min_len, "--eval_dataset_type=val"])
        print(stop)
# greedy
# subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
# subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
# subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

# subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
# subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
# subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

sys.exit()
