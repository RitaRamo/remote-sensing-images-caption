import os
import psutil
import re
import subprocess
import sys
import time

list_files = ['split_fine_encdec_1comp_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comp_eff_noaug_notnormalized.txt',
              'split_fine_encdec_13comp_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comphdsim_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comphdsimd1_eff_noaug_notnormalized.txt',
              'split_fine_encdec_4comphdall_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comphd_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comphdsimf1_eff_noaug_notnormalized.txt',
              'split_fine_encdec_4comphd_eff_noaug_notnormalized.txt',
              'split_fine_encdec_2comphddiv2_eff_noaug_notnormalized.txt']

for eval_file in list_files:
    subprocess.Popen(["python3", "src/test_generate_sentences.py", eval_file,
                      "--decodying_type='beam'", "--n_beam=10", "--test_set"])
    subprocess.Popen(["python2", "src/test_scores_coco.py", eval_file,
                      "--decodying_type='beam'", "--n_beam=10", "--test_set"])
    subprocess.Popen(["python3", "src/test_scores_bertscore.py", eval_file,
                      "--decodying_type='beam'", "--n_beam=10", "--test_set"])

sys.exit()
