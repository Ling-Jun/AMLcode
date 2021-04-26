# ------------------------------------------------------------------------------
# This is generated from https://ml.azure.com/visualinterface/authoring/Normal/d29a5698-2487-42e8-ad82-8a1208cd1472?wsid=/subscriptions/c46a9435-c957-4e6c-a0f4-b9a597984773/resourcegroups/mlops/workspaces/mlopsdev&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
# To run this code, please install SDK by this command:
# !pip install "azure-ml-component[notebooks]" --extra-index-url https://azuremlsdktestpypi.azureedge.net/modulesdkpreview --upgrade
# More detailed guide to set up your environment: https://github.com/Azure/DesignerPrivatePreviewFeatures/blob/master/azure-ml-components/samples/setup-environment.ipynb
# ------------------------------------------------------------------------------


from azureml.core import Workspace
from azure.ml.component import Pipeline, Component, dsl


# configure aml workspace
ws = Workspace.from_config()


# get components
azureml_evaluate_model_func, azureml_remove_duplicate_rows_func, azureml_two_class_neural_network_func, azureml_select_columns_in_dataset_func, azureml_clean_missing_data_func, azureml_train_model_func, azureml_two_class_support_vector_machine_func, azureml_two_class_decision_forest_func, azureml_split_data_func, azureml_score_model_func, azureml_permutation_feature_importance_func = Component.batch_load(ws, selectors=['azureml://Evaluate Model', 'azureml://Remove Duplicate Rows', 'azureml://Two-Class Neural Network', 'azureml://Select Columns in Dataset', 'azureml://Clean Missing Data', 'azureml://Train Model', 'azureml://Two-Class Support Vector Machine', 'azureml://Two-Class Decision Forest', 'azureml://Split Data', 'azureml://Score Model', 'azureml://Permutation Feature Importance'])

# get dataset
from azureml.core import Dataset
titanic_ds = Dataset.get_by_name(ws, name='titanic_ds', version=2)


# define pipeline
@dsl.pipeline(name='TitanicDesigner', description='Pipeline created on 20210415', default_compute_target='cpu-cluster', default_datastore='workspaceblobstore')
def generated_pipeline():
    azureml_select_columns_in_dataset_0 = azureml_select_columns_in_dataset_func(
        dataset=titanic_ds,
        select_columns='[{"KeepInputDataOrder":true,"ColumnNames":["Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","PassengerId"]}]')
    azureml_select_columns_in_dataset_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_clean_missing_data_0 = azureml_clean_missing_data_func(
        dataset=azureml_select_columns_in_dataset_0.outputs.results_dataset,
        columns_to_be_cleaned='[{"KeepInputDataOrder":true,"ColumnNames":["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]}]',
        minimum_missing_value_ratio=0.0,
        maximum_missing_value_ratio=1.0,
        cleaning_mode='Custom substitution value',
        replacement_value='0',
        generate_missing_value_indicator_column=False)
    azureml_clean_missing_data_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_remove_duplicate_rows_0 = azureml_remove_duplicate_rows_func(
        dataset=azureml_clean_missing_data_0.outputs.cleaned_dataset,
        key_column_selection_filter_expression='[{"KeepInputDataOrder":true,"ColumnNames":["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]}]',
        retain_first_duplicate_row=True)
    azureml_remove_duplicate_rows_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_split_data_0 = azureml_split_data_func(
        dataset=azureml_remove_duplicate_rows_0.outputs.results_dataset,
        splitting_mode='Split Rows',
        fraction_of_rows_in_the_first_output_dataset=0.7,
        randomized_split=True,
        random_seed=1234,
        stratified_split='False')
    azureml_split_data_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_two_class_decision_forest_0 = azureml_two_class_decision_forest_func(
        create_trainer_mode='SingleParameter',
        number_of_decision_trees=8,
        maximum_depth_of_the_decision_trees=32,
        minimum_number_of_samples_per_leaf_node=1,
        resampling_method='Bagging Resampling')
    azureml_two_class_decision_forest_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_train_model_0 = azureml_train_model_func(
        dataset=azureml_split_data_0.outputs.results_dataset1,
        untrained_model=azureml_two_class_decision_forest_0.outputs.untrained_model,
        label_column='[{"KeepInputDataOrder":true,"ColumnNames":["Survived"]}]',
        model_explanations=False)
    azureml_train_model_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_score_model_0 = azureml_score_model_func(
        trained_model=azureml_train_model_0.outputs.trained_model,
        dataset=azureml_split_data_0.outputs.results_dataset2,
        append_score_columns_to_output=True)
    azureml_score_model_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_two_class_support_vector_machine_0 = azureml_two_class_support_vector_machine_func(
        create_trainer_mode='SingleParameter',
        number_of_iterations=10,
        the_value_lambda=0.001,
        normalize_the_features=True,
        random_number_seed=None)
    azureml_two_class_support_vector_machine_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_train_model_1 = azureml_train_model_func(
        dataset=azureml_split_data_0.outputs.results_dataset1,
        untrained_model=azureml_two_class_support_vector_machine_0.outputs.untrained_model,
        label_column='[{"KeepInputDataOrder":true,"ColumnNames":["Survived"]}]',
        model_explanations=False)
    azureml_train_model_1.runsettings.resource_layout.configure(node_count=1)
    
    azureml_score_model_1 = azureml_score_model_func(
        dataset=azureml_split_data_0.outputs.results_dataset2,
        trained_model=azureml_train_model_1.outputs.trained_model,
        append_score_columns_to_output=True)
    azureml_score_model_1.runsettings.resource_layout.configure(node_count=1)
    
    azureml_evaluate_model_0 = azureml_evaluate_model_func(
        scored_dataset_to_compare=azureml_score_model_1.outputs.scored_dataset,
        scored_dataset=azureml_score_model_0.outputs.scored_dataset)
    azureml_evaluate_model_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_two_class_neural_network_0 = azureml_two_class_neural_network_func(
        create_trainer_mode='SingleParameter',
        hidden_layer_specification='Fully-connected case',
        number_of_hidden_nodes='100',
        the_learning_rate=0.1,
        number_of_learning_iterations=100,
        the_momentum=0,
        shuffle_examples=True,
        random_number_seed=None)
    azureml_two_class_neural_network_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_train_model_2 = azureml_train_model_func(
        dataset=azureml_split_data_0.outputs.results_dataset1,
        untrained_model=azureml_two_class_neural_network_0.outputs.untrained_model,
        label_column='[{"KeepInputDataOrder":true,"ColumnNames":["Survived"]}]',
        model_explanations=False)
    azureml_train_model_2.runsettings.resource_layout.configure(node_count=1)
    
    azureml_score_model_2 = azureml_score_model_func(
        dataset=azureml_split_data_0.outputs.results_dataset2,
        trained_model=azureml_train_model_2.outputs.trained_model,
        append_score_columns_to_output=True)
    azureml_score_model_2.runsettings.resource_layout.configure(node_count=1)
    
    azureml_evaluate_model_1 = azureml_evaluate_model_func(
        scored_dataset=azureml_score_model_1.outputs.scored_dataset,
        scored_dataset_to_compare=azureml_score_model_2.outputs.scored_dataset)
    azureml_evaluate_model_1.runsettings.resource_layout.configure(node_count=1)
    
    azureml_evaluate_model_2 = azureml_evaluate_model_func(
        scored_dataset_to_compare=azureml_score_model_2.outputs.scored_dataset,
        scored_dataset=azureml_score_model_0.outputs.scored_dataset)
    azureml_evaluate_model_2.runsettings.resource_layout.configure(node_count=1)
    
    azureml_permutation_feature_importance_0 = azureml_permutation_feature_importance_func(
        test_data=azureml_split_data_0.outputs.results_dataset2,
        trained_model=azureml_train_model_0.outputs.trained_model,
        random_seed=1234,
        metric_for_measuring_performance='Accuracy')
    azureml_permutation_feature_importance_0.runsettings.resource_layout.configure(node_count=1)
    
    azureml_permutation_feature_importance_1 = azureml_permutation_feature_importance_func(
        test_data=azureml_split_data_0.outputs.results_dataset2,
        trained_model=azureml_train_model_1.outputs.trained_model,
        random_seed=1234,
        metric_for_measuring_performance='Accuracy')
    azureml_permutation_feature_importance_1.runsettings.resource_layout.configure(node_count=1)
    
    azureml_permutation_feature_importance_2 = azureml_permutation_feature_importance_func(
        trained_model=azureml_train_model_2.outputs.trained_model,
        test_data=azureml_split_data_0.outputs.results_dataset2,
        random_seed=1234,
        metric_for_measuring_performance='Accuracy')
    azureml_permutation_feature_importance_2.runsettings.resource_layout.configure(node_count=1)
    

# create a pipeline
pipeline = generated_pipeline()

# validate pipeline and visualize the graph
pipeline.validate()

# submit a pipeline run
pipeline.submit(experiment_name='sample-experiment-name').wait_for_completion()

print(' Experiment completed')