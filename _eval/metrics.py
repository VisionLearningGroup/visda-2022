"""Implementation of running metrics."""
import numpy as np
from sklearn.metrics import confusion_matrix 

class RunningMetric():
  def __init__(self):
    pass
  
  def update(self, ground_truth, prediction):
    gt_shape = ground_truth.shape
    if len(gt_shape) > 1:
      raise ValueError("Input labels must be a 1D array, got %s" % gt_shape)
    if gt_shape != prediction.shape:
      raise ValueError("Shape mismatch: %s and %s" % 
                       (ground_truth.shape, prediction.shape))
  
  def result(self):
    pass


class RunningmIoU(RunningMetric):
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    This code is largely based on the mIoU implementation from 
    https://github.com/warmspringwinds/pytorch-segmentation-detection/
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        super(RunningmIoU, self).update(ground_truth, prediction)
        if (ground_truth == self.ignore_label).all():
            return
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            self.overall_confusion_matrix = current_confusion_matrix
    
    def result(self):
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union

class PixelAccuracy(RunningMetric):
  def __init__(self):
    self.num_samples = 0.
    self.pixel_acc = 0.

  def update(self, ground_truth, prediction):
    super(PixelAccuracy, self).update(ground_truth, prediction)
    self.num_samples += 1.
    correct_pred = np.sum(ground_truth == prediction, dtype=np.float32)
    self.pixel_acc += correct_pred / ground_truth.shape[0]

  def result(self):
    if self.num_samples == 0:
      return 0.
    return self.pixel_acc / self.num_samples
