import os
import json
from argparse import ArgumentParser

def result_after_round(round_num, report, max_rounds):

    with open(report, 'r') as f:
        progress_report = json.load(f)

    j_and_f = []

    for seq in list(progress_report.keys()):
        round_limit = max(map(int,progress_report[seq].keys()))        
        j_and_f.append(progress_report[seq][str(min(round_num, round_limit))]["J_AND_F"])

    j_and_f_mean = sum(j_and_f)/len(j_and_f)
    return j_and_f_mean

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--round",type=int, required=True)
    parser.add_argument("--report",type=str, required=True)
    parser.add_argument("--max_rounds",type=int, required=True)
    args = parser.parse_args()
    j_and_f = result_after_round(args.round, args.report, args.max_rounds)
    print(f'J_AND_F after {args.round} rounds: {j_and_f}')