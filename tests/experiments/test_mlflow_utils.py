from src.experiments.mlflow_utils import MlflowLog
import os
import mlflow
import pytest

@pytest.fixture
def metrics_params():
  test_metrics = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
  test_params = {'date': '2022-01-01', 'ts': '1666309693'}
  return test_metrics, test_params
  
@pytest.fixture
def path_experiment():
  mlflow_path = os.getcwd()
  experiment_name = 'test_experiment'
  return mlflow_path, experiment_name

def test_log_run_metrics(metrics_params, path_experiment):

  mlflow_path, experiment_name = path_experiment
  test_metrics, _ = metrics_params

  ut_mlflow = MlflowLog(mlflow_path = mlflow_path,
                      experiment_name = experiment_name,
                      metrics = [test_metrics])
  
  ut_mlflow.log_run()
  
  experiment_id = ut_mlflow.run._info.experiment_id
  artifact_uri = ut_mlflow.run._info.artifact_uri
  
  logged_metrics = mlflow.tracking.MlflowClient().search_runs(experiment_ids=experiment_id, filter_string = f"attribute.artifact_uri = '{artifact_uri}'")[0]._data.metrics
  
  assert test_metrics == logged_metrics

def test_log_run_params(metrics_params, path_experiment):

  mlflow_path, experiment_name = path_experiment
  _, test_params = metrics_params

  ut_mlflow = MlflowLog(mlflow_path = mlflow_path,
                      experiment_name = experiment_name,
                      params = [test_params])
  
  ut_mlflow.log_run()
  
  experiment_id = ut_mlflow.run._info.experiment_id
  artifact_uri = ut_mlflow.run._info.artifact_uri
  
  logged_params = mlflow.tracking.MlflowClient().search_runs(experiment_ids=experiment_id, filter_string = f"attribute.artifact_uri = '{artifact_uri}'")[0]._data.params
  
  assert test_params == logged_params