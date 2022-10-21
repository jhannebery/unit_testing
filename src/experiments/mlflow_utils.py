import mlflow
import os
from typing import List, Dict, Any

class MlflowLog():
  """
  A class used for logging metrics, parameters, models and artifacts to MLflow
  
  Model logging currently only supports models of xgboost, sklearn or lightgbm flavour.

  ...

  Attributes
  ----------
  mlflow_path : str
      path to folder where the mlflow experiment will be logged
  experiment_name : str
      name of the experiment
  model : model object
      the model we will save in the model registry in MLflow
  metrics : list[dict]
      a list of dictionaries that have key:value pairs of metric_name:metric_value
      to be logged by MLflow
      examples include auc, f1, recall, etc
  params : list[dict]
      a list of dictionaries that have key:value pairs of parameter_name:parameter_value
      to belogged by MLflow
      examples include date, model hyperparameters, etc.
  model_name : str
      name given to the model we will save in MLflow
  model_type : str
      type of model, support right now only for 'xgboost', 'sklearn', 'lightgbm'

  Methods
  -------
  log_run()
      Starts an MLflow run to the given experiment name, and logs metric, parameters and the model if they are given
      
  promote(stage='Staging')
      Promotes the model to the given stage, default Staging
  """
  
  def __init__(self, mlflow_path: str, experiment_name: str, model: Any = None, metrics: List[Dict] = None, params: List[Dict] = None, model_name: str = None, model_type: str = None):
    """
    Supports XGBoost, LightGBM, Scikit-Learn models
    """
    self.mlflow_path = mlflow_path
    self.experiment_name = experiment_name
    self.metrics = metrics
    self.params = params
    self.model_name = model_name
    self.model = model
    self.model_type = model_type
  
  def log_run(self):
    mlflow.set_experiment(os.path.join(self.mlflow_path, self.experiment_name))
    
    with mlflow.start_run() as self.run:
      
      if self.metrics:
        for metric_dict in self.metrics:
          mlflow.log_metrics(metric_dict)
      
      if self.params:
        for param_dict in self.params:
          mlflow.log_params(param_dict)
        
      if self.model:
        if self.model_type == 'xgboost':
          mlflow.xgboost.log_model(xgb_model = self.model, artifact_path = 'model', registered_model_name = self.model_name)
        if self.model_type == 'lightgbm':
          mlflow.lightgbm.log_model(lgb_model = self.model, artifact_path = 'model', registered_model_name = self.model_name)
        if self.model_type == 'sklearn':
          mlflow.sklearn.log_model(sk_model = self.model, artifact_path = 'model', registered_model_name = self.model_name)
    
  def promote(self, stage: str ='Staging'):
    self.reg_model = mlflow.tracking.MlflowClient().get_registered_model(self.model_name)
    self.reg_ver = [ver for ver in self.reg_model.latest_versions if ver.run_id == self.run.info.run_id][0]
    mlflow.tracking.MlflowClient().transition_model_version_stage(name = self.model_name, version=self.reg_ver.version, stage=stage)

class MlflowLoad():
  """
  A class used for loading models from the model registry in MLflow for inference
  
  Currently supports only models of xgboost, sklearn or lightgbm flavour.

  ...

  Attributes
  ----------
  model_name : str
      name given to the model we will save in MLflow
  model_type : str
      type of model, support right now only for 'xgboost', 'sklearn', 'lightgbm'

  Methods
  -------
  load_latest(stage=None)
      Loads the latest model saved in the model registry for the given stage under the given model name
      Returns a model object
      
  load_by_version(version='Staging')
      Loads model that matches the given version in the model registry for the given model name
      Returns a model object
  """
  def __init__(self, model_name: str, model_type: str):
    """
    Supports XGBoost, LightGBM, Scikit-Learn models
    """
    self.model_name = model_name
    self.model_type = model_type
    if self.model_type == 'xgboost':
      self.load_func = mlflow.xgboost.load_model
    if self.model_type == 'lightgbm':
      self.load_func = mlflow.lightgbm.load_model
    if self.model_type == 'sklearn':
      self.load_func = mlflow.sklearn.load_model
      
  def load_latest(self, stage=None):
    reg_model = mlflow.tracking.MlflowClient().get_registered_model(self.model_name)
    for ver in sorted(reg_model.latest_versions, key=lambda x: x.version, reverse=True):
      if stage == None or ver.current_stage == stage:
        print(f'Loading model version: {ver.version}, trained by: {ver.user_id}')
        mlf_model = self.load_func(ver.source)
        return mlf_model
    raise ValueError(f'No models found in the stage {stage} for the model {self.model_name}')
  
  def load_by_version(self, version='1'):
    reg_model = mlflow.tracking.MlflowClient().search_model_versions("name='{}'".format(self.model_name))
    for ver in reg_model:
      if ver.version == version:
        print('Loading model ver:{}, trained by: {}'.format(ver.version, ver.user_id))
        mlf_model = self.load_func(ver.source)
        return mlf_model
    raise ValueError(f'No model found matching version {ver.version} for model {self.model_name}')