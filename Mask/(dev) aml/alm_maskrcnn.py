#%%
import os
import azureml.core
from azureml.core import Workspace, Dataset, Datastore, ComputeTarget, RunConfiguration, Experiment

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

#%%
# load workspace
ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      #'Azure region: ' + workspace.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

#%%
from azureml.core import Experiment

script_folder = './code'
os.makedirs(script_folder, exist_ok=True)

exp = Experiment(workspace=ws, name='zw_mask_rcnn_x1')

from azureml.core import Datastore
blob_datastore = Datastore.register_azure_file_share(workspace=ws, 
                                                         datastore_name='SECRET', 
                                                         file_share_name='SECRET', 
                                                         account_name='SECRET',
                                                         account_key='SECRET')

#get named datastore from current workspace
datastore = Datastore.get(ws, datastore_name='project_zero')

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
#cluster_name = "StandardNC12"
cluster_name =  'gpucluster'

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC12', 
                                                           max_nodes=1, min_nodes=1)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())


#%%
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
# set up environment
env = Environment('cv-pipeline')

cd = CondaDependencies.create(pip_packages=['numpy', 'scipy','Pillow','cython','azureml-defaults',
                                            'matplotlib', 'scikit-image', 'tensorflow==1.13.1', 'keras>=2.0.8','opencv-python',
                                            'h5py', 'imgaug', 'IPython[all]'])
env.python.conda_dependencies = cd

#%%
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': datastore.as_mount()
}


est = Estimator(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target, 
                 entry_script='alm_trainer.py', 
                 environment_definition= env)

#%%
run = exp.submit(est)
run


#%%
#from azureml.widgets import RunDetails
#RunDetails(run).show()


#%%
# Shows output of the run on stdout.
#run.wait_for_completion(show_output=True)