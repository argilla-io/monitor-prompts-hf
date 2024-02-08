import os
from typing import Dict, Tuple
from uuid import UUID

import argilla as rg
from argilla.feedback import FeedbackDataset
from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset
import gradio as gr

# Connect to the space with rg.init()
rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    extra_headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
)


def obtain_source_target_datasets() -> (
    Tuple[
        FeedbackDataset | RemoteFeedbackDataset, FeedbackDataset | RemoteFeedbackDataset
    ]
):
    """
    This function returns the source and target datasets to be used in the application.

    Returns:
        A tuple with the source and target datasets. The source dataset is filtered by the response status 'pending'.

    """

    # Obtain the public dataset and see how many pending records are there
    source_dataset = rg.FeedbackDataset.from_argilla(
        os.getenv("SOURCE_DATASET"), workspace=os.getenv("SOURCE_WORKSPACE")
    )
    filtered_source_dataset = source_dataset.filter_by(response_status=["pending"])

    # Obtain a list of users from the private workspace
    target_dataset = rg.FeedbackDataset.from_argilla(
        os.getenv("RESULTS_DATASET"), workspace=os.getenv("RESULTS_WORKSPACE")
    )

    return filtered_source_dataset, target_dataset


def get_user_annotations_dictionary(
    dataset: FeedbackDataset | RemoteFeedbackDataset,
) -> Dict[str, int]:
    """
    This function returns a dictionary with the username as the key and the number of annotations as the value.

    Args:
        dataset: The dataset to be analyzed.
    Returns:
        A dictionary with the username as the key and the number of annotations as the value.
    """
    output = {}
    for record in dataset:
        for response in record.responses:
            if str(response.user_id) not in output.keys():
                output[str(response.user_id)] = 1
            else:
                output[str(response.user_id)] += 1

    # Changing the name of the keys, from the id to the username
    for key in list(output.keys()):
        output[rg.User.from_id(UUID(key)).username] = output.pop(key)

    return output

source_dataset, target_dataset = obtain_source_target_datasets()
user_ids_annotations = get_user_annotations_dictionary(target_dataset)
