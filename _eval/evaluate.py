import numpy as np
import os, glob
import tqdm
import zipfile
import imageio
import metrics
from pathlib import Path
import shutil


opj = os.path.join
# change this global variable to the evalAI temp path
TEMP_ROOT = "/scratch/dinka/data/zerowaste_eval_tmp"

def unzip_to_folder(zip_path, out_folder_path):
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    os.makedirs(out_folder_path, exist_ok=True)
    zip_ref.extractall(out_folder_path)


def evaluate(
    test_annotation_file,
    user_submission_file,
    phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:
        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made
        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']
        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    temp_pred_dir = os.path.join(
        TEMP_ROOT, "%s_%i_%s_%s" % (
            kwargs['submission_metadata']["participant_team_name"],
            kwargs['submission_metadata']["id"],
            phase_codename, 
            kwargs['submission_metadata']["submitted_at"]))
    rand_id = np.random.randint(0, 10000)
    temp_gt_dir = os.path.join(TEMP_ROOT,  "gt_%05d" % rand_id)
    unzip_to_folder(test_annotation_file, temp_gt_dir)
    unzip_to_folder(user_submission_file, temp_pred_dir)
    assert np.all([x in os.listdir(temp_gt_dir) for x in ["val", "test"]])
    gt_val_list = list(Path(opj(temp_gt_dir, "val")).rglob("*.[pP][nN][gG]"))
    gt_test_list = list(Path(opj(temp_gt_dir, "test")).rglob("*.[pP][nN][gG]"))

    if phase_codename == "val":
      gt_list = gt_val_list
    else:
      gt_list = gt_test_list
    if not "source_only" in os.listdir(temp_pred_dir):
      raise FileNotFoundError("source_only folder not found in archive.")
    if not "uda" in os.listdir(temp_pred_dir):
      raise FileNotFoundError("uda folder not found in archive.")
    
    preds_so_list = list(Path(opj(temp_pred_dir, "source_only")).rglob("*.[pP][nN][gG]"))
    preds_uda_list = list(Path(opj(temp_pred_dir, "uda")).rglob("*.[pP][nN][gG]"))
    pred_metrics = { 
        "source_only": { 
        "mIoU": metrics.RunningmIoU(labels=range(5)),
        "Acc": metrics.PixelAccuracy()
    },
    "uda": { 
        "mIoU": metrics.RunningmIoU(labels=range(5)),
        "Acc": metrics.PixelAccuracy()
        }
    }
    for gt_seg in tqdm.tqdm(gt_list):
      bname = os.path.basename(gt_seg)
      gt_img = imageio.imread(gt_seg)
      flat_gt = gt_img.reshape(-1)
      for method in ["source_only", "uda"]:
        if not os.path.exists(opj(temp_pred_dir, method, bname)):
          raise FileNotFoundError("Prediction not found for %s of %s" % (bname, method))
        pred_img = imageio.imread(opj(temp_pred_dir, method, bname))
        # remove the following line in the final version
        flat_pred = pred_img.reshape(-1)
        for mtr in ["mIoU", "Acc"]:
          pred_metrics[method][mtr].update(
              ground_truth=flat_gt, prediction=flat_pred)

    output = {}

    print("Evaluating for %s phase" % phase_codename)
    output["result"] = [
        {
          "%s_source_only" % phase_codename: {
            x: pred_metrics["source_only"][x].result() for x in pred_metrics["source_only"]},
          "%s_uda" % phase_codename: {
            x: pred_metrics["uda"][x].result() for x in pred_metrics["uda"]},
        }
    ]
    # To display the results in the result file
    print("Completed evaluation for %s Phase" % phase_codename)
    shutil.rmtree(temp_gt_dir)
    return output