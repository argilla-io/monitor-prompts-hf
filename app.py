import os
from typing import Dict, Tuple
from uuid import UUID

import altair as alt
import argilla as rg
from argilla.feedback import FeedbackDataset
from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset
import gradio as gr
import pandas as pd


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


def donut_chart() -> alt.Chart:
    """
    This function returns a donut chart with the number of annotated and pending records.

    Returns:
        An altair chart with the donut chart.
    """

    source_dataset, _ = obtain_source_target_datasets()
    annotated_records = len(source_dataset)
    pending_records = int(os.getenv("TARGET_RECORDS")) - annotated_records

    source = pd.DataFrame(
        {
            "values": [annotated_records, pending_records],
            "category": ["Annotated", "Pending"],  # Add a new column for categories
        }
    )

    base = alt.Chart(source).encode(
        theta=alt.Theta("values:Q", stack=True),
        radius=alt.Radius(
            "values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)
        ),
        color=alt.Color("category:N", legend=alt.Legend(title="Category")),
    )

    c1 = base.mark_arc(innerRadius=20, stroke="#fff")

    c2 = base.mark_text(radiusOffset=10).encode(text="values:Q")

    chart = c1 + c2

    return chart


def kpi_chart() -> alt.Chart:
    """
    This function returns a KPI chart with the total amount of annotators.

    Returns:
        An altair chart with the KPI chart.
    """

    # Obtain the total amount of annotators
    _, target_dataset = obtain_source_target_datasets()
    user_ids_annotations = get_user_annotations_dictionary(target_dataset)
    total_annotators = len(user_ids_annotations)

    # Assuming you have a DataFrame with user data, create a sample DataFrame
    data = pd.DataFrame({"Category": ["Total Annotators"], "Value": [total_annotators]})

    # Create Altair chart
    chart = (
        alt.Chart(data)
        .mark_text(fontSize=100, align="center", baseline="middle", color="steelblue")
        .encode(text="Value:N")
        .properties(title="Number of Annotators", width=250, height=200)
    )

    return chart


def obtain_top_5_users(user_ids_annotations: Dict[str, int]) -> pd.DataFrame:
    """
    This function returns the top 5 users with the most annotations.

    Args:
        user_ids_annotations: A dictionary with the user ids as the key and the number of annotations as the value.

    Returns:
        A pandas dataframe with the top 5 users with the most annotations.
    """

    dataframe = pd.DataFrame(
        user_ids_annotations.items(), columns=["Name", "Annotated Records"]
    )
    dataframe = dataframe.sort_values(by="Annotated Records", ascending=False)
    return dataframe.head(5)


def main() -> None:

    # Connect to the space with rg.init()
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
        extra_headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
    )

    source_dataset, target_dataset = obtain_source_target_datasets()
    user_ids_annotations = get_user_annotations_dictionary(target_dataset)

    top5_dataframe = obtain_top_5_users(user_ids_annotations)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🗣️ The Prompt Collective Dashboad

            This Gradio dashboard shows the progress of the first "Data is Better Together" initiative to understand and collect good quality and diverse prompt for the OSS AI community.
            If you want to contribute to OSS AI, join [the Prompt Collective HF Space](https://huggingface.co/spaces/DIBT/prompt-collective).
            """
        )
        gr.Markdown(
            """
            ## 🚀 Contributors Progress

            How many records have been submitted, how many are still pending?
            """
        )
        plot = gr.Plot(label="Plot")
        demo.load(
            donut_chart,
            inputs=[],
            outputs=[plot],
        )

        gr.Markdown(
            """
            ## 👾 Contributors Hall of Fame
            The number of all annotators and the top 5 users with the most responses are:
            """
        )

        with gr.Row():

            plot2 = gr.Plot(label="Plot")
            demo.load(
                kpi_chart,
                inputs=[],
                outputs=[plot2],
            )

            gr.Dataframe(
                value=top5_dataframe,
                headers=["Name", "Annotated Records"],
                datatype=[
                    "str",
                    "number",
                ],
                row_count=5,
                col_count=(2, "fixed"),
                interactive=False,
            ),

    # Launch the Gradio interface
    demo.launch()


if __name__ == "__main__":
    main()
