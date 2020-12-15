import sys
import subprocess

eval_file = sys.argv[1]

print("File that will be evaluated", eval_file)

subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0"])

subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0"])

subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0"])

subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                        "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0",
                        "--eval_dataset_type=val"])

subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0",
                "--eval_dataset_type=val"])

subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam_wt_refinement", "--n_beam=3", "--min_len=0", "--rep_window=0",
                "--eval_dataset_type=val"])

sys.exit()
