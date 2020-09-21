import os
import psutil
import re
import subprocess
import sys
import time

list_files = ['ucml_fine_encdec_3comp_eff_noaug_notnormalized.txt',
              'xasa.txt']

for eval_file in list_files:
    subprocess.run(["python3", "src/test_generate_sentences.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=10"])
    subprocess.run(["python2", "src/test_scores_coco.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=10"])
    subprocess.run(["python3", "src/test_scores_bertscore.py", "@experiments/conf_files/" + eval_file,
                    "--decodying_type=beam", "--n_beam=10"])

sys.exit()
