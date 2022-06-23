from typing import Dict
from typing import List

from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDescriptor,
)


class DescriptionScreen:
    default_args = {
        "html_primary_caption_size": 16,
        "html_caption_size": 14,
        "html_normal_size": 12,
        "html_attribute_caption_size": 13,
    }

    static_attribute_description = (
        "Case level attributes are defined on the case "
        "level. They don't change over "
        "the course of a case."
    )
    dynamic_attribute_description = (
        "Activity level attributes are defined on the "
        "activity level. They stay constant over the "
        "course of a case"
    )

    def __init__(
        self,
        analysis_name: str,
        analysis_goal: str,
        analysis_definition: str,
        static_attribute_descriptors: List[AttributeDescriptor],
        dynamic_attribute_descriptors: List[AttributeDescriptor],
        **kwargs,
    ):
        self.analysis_name = analysis_name
        self.analysis_goal = analysis_goal
        self.analysis_definition = analysis_definition
        self.static_attribute_descriptors = static_attribute_descriptors
        self.dynamic_attribute_descriptors = dynamic_attribute_descriptors
        self.html_args = self._get_html_vars(**kwargs)
        self.description_box = VBox()  # The box that is the view

    def _get_html_vars(self, **kwargs) -> Dict[str, str]:
        """Get html styling arguments. If an argument is provided in kwargs, that one
        will be used, else the one from self.default_args will be used.

        :param kwargs: arguments
        :return: ditionary with html arguments
        """
        html_args = {}
        html_args["html_caption_size"] = (
            self.default_args["html_caption_size"]
            if kwargs.get("caption_size") is None
            else kwargs.get("html_caption_size")
        )
        html_args["html_normal_size"] = (
            self.default_args["html_normal_size"]
            if kwargs.get("html_normal_size") is None
            else kwargs.get("html_normal_size")
        )
        html_args["html_attribute_caption_size"] = (
            self.default_args["html_attribute_caption_size"]
            if kwargs.get("html_attribute_caption_size") is None
            else kwargs.get("html_attribute_caption_size")
        )
        html_args["html_primary_caption_size"] = (
            self.default_args["html_primary_caption_size"]
            if kwargs.get("html_primary_caption_size") is None
            else kwargs.get("html_primary_caption_size")
        )
        return html_args

    def create_description_screen(self):
        """Populate the description screen"""
        analysis_description = self._create_analysis_description()
        attribute_description = self._create_atrributes_descriptions_box()
        self.description_box.children = [analysis_description, attribute_description]

    def _create_analysis_description(self) -> HTML:
        """Create an HTML object with the description of the analysis.

        :return: HTML object
        """
        html_header_str = (
            f'<div style="line-height:140%;font-weight:bold; font-size: '
            f'{self.html_args["html_primary_caption_size"]}px"><p> '
            f"{self.analysis_name}</p></div>"
        )
        html_goal_str = (
            f'<div style="margin-top: 10px;line-height:140%;font-weight:bold; '
            f"font-size: "
            f'{self.html_args["html_caption_size"]}px"><p>Goal '
            f"</p></div><div "
            f'style="line-height:140%;font-size: '
            f'{self.html_args["html_normal_size"]}px"><p>'
            f"{self.analysis_goal}</div></p>"
        )
        html_definition_str = (
            f'<div style="margin-top: 10px;line-height:140%;font-weight:bold; '
            f"font-size: "
            f'{self.html_args["html_caption_size"]}px"><p>Definition '
            f"</p></div><div "
            f'style="line-height:140%;font-size: '
            f'{self.html_args["html_normal_size"]}px"><p>'
            f"{self.analysis_definition}</div></p>"
        )

        html_description = HTML(html_header_str + html_goal_str + html_definition_str)
        return html_description

    def _create_atrributes_descriptions_box(self) -> VBox:
        """Create an HTML object with the descriptions of the attributes.

        :return: HTML object
        """
        html_caption_str = (
            f'<div style="margin-top: 10px; margin-botton: 5px; font-weight:bold; '
            f"font-size: "
            f'{self.html_args["html_caption_size"]}px">Attribute '
            f"Descriptions</div>"
        )
        html_caption = HTML(html_caption_str)
        children_vbox = [html_caption]
        if len(self.static_attribute_descriptors) > 0:
            static_description = self._create_atrributes_description_static_dynamic(
                "static"
            )
            children_vbox.append(static_description)
        if len(self.dynamic_attribute_descriptors) > 0:
            dynamic_description = self._create_atrributes_description_static_dynamic(
                "dynamic"
            )
            children_vbox.append(dynamic_description)

        return VBox(children=children_vbox)

    def _create_atrributes_description_static_dynamic(self, attr_type: str) -> VBox:
        """Create an HTML object with the descriptions of the static attributes.

        :param attr_type: type of the attribute. either 'static' or 'dynamic
        :return: html string or empty string if there are no attribute descriptors of
        the specified type
        """
        if attr_type == "static":
            descriptors = self.static_attribute_descriptors
            html_caption_str = (
                f'<div style="line-height:140%; font-weight:bold; font-size: '
                f'{self.html_args["html_attribute_caption_size"]}px '
                f'">Case level '
                f"Attributes:</div>"
            )
            html_general_description_str = (
                f'<div style="line-height:140%; font-size: '
                f'{self.html_args["html_normal_size"]}px margin-bottom: 10px">'
                f"{self.static_attribute_description}"
                f"</div>"
            )
        elif attr_type == "dynamic":
            descriptors = self.dynamic_attribute_descriptors
            html_caption_str = (
                f'<div style="line-height:140%; font-weight:bold; font-size: '
                f'{self.html_args["html_attribute_caption_size"]}px '
                f'padding-bottom: 30px">Activity level '
                f"Attributes</div>"
            )
            html_general_description_str = (
                f'<div style="line-height:140%; font-size: '
                f'{self.html_args["html_normal_size"]}px '
                f'margin-bottom: 30px">'
                f"{self.dynamic_attribute_description}"
                f"</div>"
            )
        else:
            raise ValueError("attr_type must be either 'static' or 'dynamic'")

        attribute_descriptions = ""
        is_first = True
        for attribute_descriptor in descriptors:
            margin_bottom = "0px" if is_first else "5px"
            is_first = False
            attr_name = attribute_descriptor.display_name
            attr_description = attribute_descriptor.description
            attr_caption_html_str = (
                f'<div style="margin-top: {margin_bottom}; line-height:140%; '
                f"font-weight:bold; font-size: "
                f'{self.html_args["html_attribute_caption_size"]}px">'
                f"{attr_name}</div>"
            )
            attr_description_html_str = (
                f'<div style="line-height:140%; '
                f"font-size: "
                f'{self.html_args["html_normal_size"]}px">'
                f"{attr_description}</div>"
            )
            curr_attribute_description = (
                attr_caption_html_str + attr_description_html_str
            )
            attribute_descriptions = attribute_descriptions + curr_attribute_description

        caption_html = HTML(html_caption_str + html_general_description_str)
        attributes_html = HTML(attribute_descriptions)
        margin_static = "0px 0px 10px 0px"
        margin_dynamic = "0ox 0px 0px 0px"
        if attr_type == "static":
            margin = margin_static
        else:
            margin = margin_dynamic
        layout_attributes = Layout(border="3px double CornflowerBlue", margin=margin)
        vbox_attributes = VBox(children=[attributes_html], layout=layout_attributes)
        vbox_complete = VBox(children=[caption_html, vbox_attributes])
        return vbox_complete
