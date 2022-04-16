from typing import List

from ipywidgets import HBox
from ipywidgets import Layout
from ipywidgets import VBox
from ipywidgets import Widget
from plotly.graph_objs import FigureWidget

from one_click_analysis.feature_processing.feature_processor import FeatureProcessor
from one_click_analysis.gui.figures import AttributeDevelopmentFigure
from one_click_analysis.gui.figures import DistributionFigure
from one_click_analysis.gui.figures import SingleValueBox


class OverviewScreen:
    def create_box(self, traits: List[List[Widget]]):
        """
        :param traits: List of lists with the traits to display. The traits in an
        inner list are put into an HBox. The resulting HBoxes will be put into a VBox.
        :return:
        """
        vBox_overview_layout = Layout(border="2px solid gray", grid_gap="30px")
        vbox_overview = VBox(layout=vBox_overview_layout)
        boxes_traits = []
        for trait in traits:
            if len(trait) == 1 and isinstance(trait[0], FigureWidget):
                child = trait[0]
            else:
                child = HBox(children=trait)
            boxes_traits.append(child)
        vbox_overview.children = boxes_traits
        return vbox_overview


class OverviewScreenCaseDuration(OverviewScreen):
    def __init__(self, fp: FeatureProcessor):
        self.fp = fp

        self.overview_box = self._create_overview_screen()

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        label_column_name = self.fp.labels[0].df_attribute_name
        avg_case_duration = round(self.fp.df[label_column_name].mean(), 2)
        unit = self.fp.labels[0].unit
        # Case duration
        title = "Average case duration"
        avg_case_duration_box = SingleValueBox(
            title=title,
            val=avg_case_duration,
            unit=unit,
            title_color="Blue",
            val_color="Black",
        )
        metrics_box = HBox(
            children=[avg_case_duration_box.box],
            layout=Layout(margin="0px 30px 0px " "0px"),
        )

        # development of case duration
        fig_case_duration_development = AttributeDevelopmentFigure(
            df=self.fp.df,
            time_col="Case start time",
            attribute_cols=self.fp.labels[0].df_attribute_name,
            fill=True,
            title="Case duration development",
        )

        # case duration distribution
        fig_distribution = DistributionFigure(
            df=self.fp.df,
            attribute_col=self.fp.labels[0].df_attribute_name,
            attribute_name="Case duration",
            num_bins=10,
        )

        return self.create_box(
            [
                [metrics_box],
                [fig_case_duration_development.figure],
                [fig_distribution.figure],
            ]
        )
