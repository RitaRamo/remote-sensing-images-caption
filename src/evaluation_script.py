import sys
import subprocess


eval_file = sys.argv[1]

print("File that will be evaluated", eval_file)

for n_beam in [1, 5, 15]:
    print("n_beam", n_beam)

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=" + n_beam])
    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=10" + n_beam])
    subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=10" + n_beam])

    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])
    subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=" + n_beam, "--eval_dataset_type=val"])

# greedy
subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file])
subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file])
subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file])

sys.exit()
