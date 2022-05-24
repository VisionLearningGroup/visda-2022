import os
import argparse
import evaluate as ev

# on GEOFFREY

def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluate the submission.')
    parser.add_argument('gt_file_path',  type=str, 
                        help='path to the zip file containing ground truth labels. '
                        'This archive must contain two folders: `val` and `test`.')
    parser.add_argument('pred_file_path',  type=str, 
                        help='path to the zip file containing predicted labels. '
                        'This archive must contain two folders: `source_only` and `uda`.')
    parser.add_argument('phase', type=str, default="val",
                        help='evaluation phase: val | test')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    default_metadata = {
        "participant_team_name": "superteam",
        "id": 132,
        "submitted_at": '2017-03-20T19:22:03.880652Z'
    }
    results = ev.evaluate(
        test_annotation_file=args.gt_file_path, 
        user_submission_file=args.pred_file_path,
        phase_codename=args.phase, 
        submission_metadata=default_metadata)
    print(results)



if __name__ == "__main__":
    main()