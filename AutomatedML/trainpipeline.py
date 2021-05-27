import azureml.core
from azureml.core import Workspace
from azureml.core import Keyvault
import os

from azureml.core import Workspace, Experiment

from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Experiment

from azureml.core import Workspace
import urllib
from zipfile import ZipFile

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment 
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
from azureml.core import Workspace, Datastore
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails
from azureml.pipeline.core.graph import PipelineParameter

print("SDK version:", azureml.core.VERSION)

import argparse 
import json
import os

parse = argparse.ArgumentParser()
parse.add_argument("--tenantid")
parse.add_argument("--acclientid")
parse.add_argument("--accsecret")
    
args = parse.parse_args()


sp = ServicePrincipalAuthentication(tenant_id=args.tenantid, # tenantID
                                    service_principal_id=args.acclientid, # clientId
                                    service_principal_password=args.accsecret) # clientSecret

ws = Workspace.get(name="mlopsdev",
                   auth=sp,
                   subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773", resource_group="mlops")

#ws = Workspace.from_config()
keyvault = ws.get_default_keyvault()
tenantid = keyvault.get_secret(name="tenantid")
acclientid = keyvault.get_secret(name="acclientid")
accsvcname = keyvault.get_secret(name="accsvcname")
accsecret = keyvault.get_secret(name="accsecret")

print(accsvcname)

sp = ServicePrincipalAuthentication(tenant_id=tenantid, # tenantID
                                    service_principal_id=acclientid, # clientId
                                    service_principal_password=accsecret) # clientSecret

ws = Workspace.get(name="mlopsdev",
                   auth=sp,
                   subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773", resource_group="mlops")

ws.get_details()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')


compute_name = "cpu-cluster"
vm_size = "STANDARD_NC6"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                min_nodes=0,
                                                                max_nodes=4)
    # create the compute target
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current cluster status, use the 'status' property
    print(compute_target.status.serialize())

aml_run_config = RunConfiguration()
# `compute_target` as defined in "Azure Machine Learning compute" section above
aml_run_config.target = compute_target

USE_CURATED_ENV = True
if USE_CURATED_ENV :
    curated_environment = Environment.get(workspace=ws, name="AzureML-Tutorial")
    aml_run_config.environment = curated_environment
else:
    aml_run_config.environment.python.user_managed_dependencies = False
    
    # Add some packages relied on by data prep step
    aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=['pandas','scikit-learn','seaborn','tqdm'], 
        pip_packages=['azureml-sdk', 'azureml-dataprep[fuse,pandas]','seaborn','tqdm'], 
        pin_sdk_version=False)

web_path ='https://dprepdata.blob.core.windows.net/demo/Titanic.csv'
my_dataset = Dataset.Tabular.from_delimited_files(path=web_path, set_column_types={'Survived': DataType.to_bool()})

dataprep_source_dir = "./dataprep_src"
#entry_point = "prepare.py"
# `my_dataset` as defined above
ds_input = my_dataset.as_named_input('input1')

datastore = ws.get_default_datastore()

output_data1 = OutputFileDatasetConfig(destination = (datastore, 'outputdataset/{run-id}'))
output_data_dataset = output_data1.register_on_complete(name = 'titanic_output_data')

train_source_dir = "./train_src"
train_entry_point = "train.py"

training_results = OutputFileDatasetConfig(name = "training_results",
    destination = def_blob_store)

    
train_step = PythonScriptStep(
    script_name=train_entry_point,
    source_directory=train_source_dir,
    arguments=["--input_data", ds_input],
    compute_target=compute_target, # , "--training_results", training_results
    runconfig=aml_run_config,
    allow_reuse=False
)

compare_models = [train_step]


# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=train_step)

pipeline1.validate()
print("Pipeline validation complete")

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'Titanic_Pipeline_Notebook').submit(pipeline1)
pipeline_run1.wait_for_completion()

RunDetails(pipeline_run1).show()

step_runs = pipeline_run1.get_children()
for step_run in step_runs:
    status = step_run.get_status()
    print('Script:', step_run.name, 'status:', status)
    
    # Change this if you want to see details even if the Step has succeeded.
    if status == "Failed":
        joblog = step_run.get_job_log()
        print('job log:', joblog)


pipeline_param = PipelineParameter(
  name="pipeline_arg",
  default_value=10)

published_pipeline1 = pipeline_run1.publish_pipeline(
     name="Published_Titanic_Pipeline_Notebook",
     description="Titanic_Pipeline_Notebook Published Pipeline Description",
     version="1.0")

from azureml.pipeline.core import PublishedPipeline
import requests

response = requests.post(published_pipeline1.endpoint, 
                         json={"ExperimentName": "Titanic_Pipeline_Notebook",
                               "ParameterAssignments": {"pipeline_arg": 20}})

from azureml.pipeline.core import PipelineEndpoint

published_pipeline = PipelineEndpoint.get(workspace=ws, name="Published_Titanic_Pipeline_Notebook")
pipeline_endpoint = PipelineEndpoint.publish(workspace=ws, name="TitanicPipelineEndpointTest",
                                            pipeline=published_pipeline, description="Test Published_Titanic_Pipeline_Notebook description Notebook")

pipeline_endpoint_by_name = PipelineEndpoint.get(workspace=ws, name="Published_Titanic_Pipeline_Notebook")
run_id = pipeline_endpoint_by_name.submit("PipelineEndpointExperiment")
print(run_id)