from multiprocessing.spawn import prepare
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from matplotlib.colors import to_rgba

### PLOTS

def prepare_colors():
    with open("data/config.yaml") as file:
        config = yaml.safe_load(file)

    colors = config["plotting"]["tech_colors"]

    colors["electricity grid"] = colors["electricity"]
    colors["ground-sourced ambient"] = colors["ground heat pump"]
    colors["air-sourced ambient"] = colors["air heat pump"]
    colors["co2 atmosphere"] = colors["co2"]
    colors["co2 stored"] = colors["co2"]
    colors["net co2 emissions"] = colors["co2"]
    colors["co2 sequestration"] = colors["co2"]
    colors["fossil oil"] = colors["oil"]
    colors["fossil gas"] = colors["gas"]
    colors["biogas to gas"] = colors["biogas"]
    colors["process emissions from feedstocks"] = colors["process emissions"]

    gas_boilers = ['residential rural gas boiler', 'services rural gas boiler',
        'residential urban decentral gas boiler',
        'services urban decentral gas boiler', 'urban central gas boiler']
    for gas_boiler in gas_boilers:
        colors[gas_boiler] = colors["gas boiler"]

    colors["urban central gas CHP"] = colors["CHP"]
    colors["urban central gas CHP CC"] = colors["CHP"]
    colors["urban central solid biomass CHP"] = colors["CHP"]
    colors["urban central solid biomass CHP CC"] = colors["CHP"]
    
    return colors


def plot_sankey(connections):

    labels = np.unique(connections[["source", "target"]])

    nodes = pd.Series({v: i for i, v in enumerate(labels)})

    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

    link_colors = [
        "rgba{}".format(to_rgba(node_colors[src], alpha=0.5))
        for src in connections.source
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",  # [snap, nodepad, perpendicular, fixed]
            valuesuffix="TWh",
            valueformat=".1f",
            node=dict(pad=10, thickness=5, label=nodes.index, color=node_colors),
            link=dict(
                source=connections.source.map(nodes),
                target=connections.target.map(nodes),
                value=connections.value,
                label=connections.label,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(height=800)

    return fig


def plot_carbon_sankey(co2):

    labels = np.unique(co2[["source", "target"]])

    nodes = pd.Series({v: i for i, v in enumerate(labels)})
    
    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

    link_colors = [
        "rgba{}".format(to_rgba(colors[src], alpha=0.5))
        for src in co2.label
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="freeform",  # [snap, nodepad, perpendicular, fixed]
            valuesuffix=" MtCO2",
            valueformat=".1f",
            node=dict(
                pad=5,
                thickness=5,
                label=nodes.index,
                color=node_colors
            ),
            link=dict(
                source=co2.source.map(nodes),
                target=co2.target.map(nodes),
                value=co2.value,
                label=co2.label,
                color=link_colors
            ),
        )
    )

    fig.update_layout(height=800)

    return fig


### MAIN

colors = prepare_colors()

## DISPLAY
st. set_page_config(layout="wide")

with st.sidebar:
    st.title("Benefits of a Hydrogen Network in Europe")

    st.markdown("""
        Explore trade-offs between electricity grid reinforcements and building
        a new hydrogen backbone with repurposed gas pipelines in the European
        energy system.
    """)

    choices = {"electricity": "yes", "no-electricity": "no"}
    power_grid = st.selectbox("Electricity network expansion", choices, format_func=lambda x: choices[x])

    choices = {"hydrogen": "yes", "no-hydrogen": "no"}
    hydrogen_grid = st.selectbox("Hydrogen network expansion", choices, format_func=lambda x: choices[x])

    display = st.radio("Explore scenarios", ["System cost comparison", "Spatial configurations", "System operation", "Sankey of energy flows", "Sankey of carbon flows"])

    # with st.expander("PyPSA-Eur-Sec"):
    #     st.write("""
    #         All results were created using the open European energy system model PyPSA-Eur-Sec.
    #         The model covers all energy sectors including electricity, buildings, transport, agriculture and industry
    #         at high spatio-temporal resolution. The model code is available on
    #         [Github](http://github.com/pypsa/pypsa-eur-sec).
    #     """)

    link = '[Read the paper](http://arxiv.org/TBA)'
    st.markdown(link, unsafe_allow_html=True)

    link = '[Download the data](http://zenodo.com/TBA)'
    st.markdown(link, unsafe_allow_html=True)

    link = '[Inspect the code](http://github.com/fneum/spatial-sector)'
    st.markdown(link, unsafe_allow_html=True)


if display == "System cost comparison":

    st.image("data/graphical-abstract.png")

if display == "System operation":

    st.warning("Coming soon!")

if display == "Spatial configurations":

    st.warning("Coming soon!")

if display == "Sankey of carbon flows":

    st.write("""
    # Carbon Sankey
    """)

    df = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/sankey-carbon.csv")

    st.plotly_chart(plot_carbon_sankey(df), use_container_width=True)


if display == "Sankey of energy flows":

    st.write("""
    # Energy Sankey
    """)

    df = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/sankey.csv")

    st.plotly_chart(plot_sankey(df), use_container_width=True)

