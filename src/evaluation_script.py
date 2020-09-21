import os
import sys

eval_file = sys.argv[0]
print("File that will be evaluated", eval_file)

subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10"])
subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10"])
subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10"])

subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10", "--eval_dataset_type=val"])
subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10", "--eval_dataset_type=val"])
subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                "--decodying_type=beam", "--n_beam=10", "--eval_dataset_type=val"])

sys.exit()
