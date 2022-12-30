import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from matplotlib.colors import to_rgba

import geopandas as gpd
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import pypsa
import datetime
import hvplot.pandas

from helpers import prepare_colors, rename_techs_tyndp

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
            valuesuffix=" TWh",
            valueformat=".1f",
            node=dict(pad=4, thickness=10, label=nodes.index, color=node_colors),
            link=dict(
                source=connections.source.map(nodes),
                target=connections.target.map(nodes),
                value=connections.value,
                label=connections.label,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=20, t=0, b=0)
    )

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
                pad=4,
                thickness=10,
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

    fig.update_layout(
        height=800, 
        margin=dict(l=100, r=0, t=0, b=150)
    )

    return fig


@st.cache
def nodal_balance(power_grid, hydrogen_grid, carrier):

    df = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/balance-ts-{carrier}.csv", index_col=0, parse_dates=True)

    df = df.groupby(df.columns.map(rename_techs_tyndp), axis=1).sum()

    df = df.loc[:, ~df.columns.isin(["H2 pipeline", "transmission lines"])]

    missing = df.columns.difference(preferred_order)
    order = preferred_order.intersection(df.columns).append(missing)
    df = df.loc[:, order]
    
    return df

@st.cache
def load_report(power_grid, hydrogen_grid):
    return pd.concat([
        pd.read_csv("data/report.csv", index_col=0, header=[0,1]),
        pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/report.csv", index_col=0, header=[0,1])
    ], axis=1)

@st.cache(allow_output_mutation=True)
def load_regions():
    fn = "../../papers/spatial-sector/workflows-rev0/pypsa-eur/resources/regions_onshore_elec_s_181.geojson"
    gdf = gpd.read_file(fn).set_index('name')
    gdf['name'] = gdf.index
    return gdf


@st.cache(allow_output_mutation=True)
def load_network():
    return pypsa.Network("../../papers/spatial-sector/workflows-rev0/pypsa-eur-sec/results/20211218-181-h2/postnetworks/elec_s_181_lvopt__Co2L0-3H-T-H-B-I-A-solar+p3-linemaxext10_2030.nc")


@st.cache(allow_output_mutation=True)
def make_electricity_graph():

    n = load_network()

    rename_link_attrs = {'p_nom': 's_nom', 'p_nom_opt': 's_nom_opt'}

    edges = pd.concat([
        n.lines,
        n.links.loc[n.links.carrier=='DC'].rename(rename_link_attrs, axis=1),
    ], axis=0)

    edges["Total Capacity (GW)"] = edges.s_nom_opt.div(1e3)
    edges["Reinforcement (GW)"] = (edges.s_nom_opt - edges.s_nom).div(1e3)
    edges["Original Capacity (GW)"] = edges.s_nom.div(1e3)
    edges["Technology"] = edges.carrier

    attr = ["Total Capacity (GW)", "Reinforcement (GW)", "Original Capacity (GW)", "Technology"]
    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)

    pos = pd.concat([n.buses.x, n.buses.y], axis=1).apply(tuple, axis=1).to_dict()

    return G, pos

@st.cache(allow_output_mutation=True)
def make_hydrogen_graph():

    n = load_network()

    edges = n.links.query("carrier == 'H2 pipeline'")

    edges["Total Capacity (GW)"] = edges.p_nom_opt.div(1e3)

    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=["Total Capacity (GW)"])

    pos = pd.concat([n.buses.x, n.buses.y], axis=1).apply(tuple, axis=1).to_dict()

    return G, pos

def parse_spatial_options(x):
    return " - ".join(x) if x != 'Nothing' else 'Nothing'

### MAIN

with open("data/config.yaml") as file:
    config = yaml.safe_load(file)

colors = prepare_colors(config)

preferred_order = pd.Index(config['preferred_order'])

## DISPLAY
st.set_page_config(page_title='European Hydrogen Network', layout="wide")

st.write('<style>div.block-container{padding-top:.5rem; padding-bottom:0rem; padding-right:1.2rem; padding-left:1.2rem}</style>', unsafe_allow_html=True)

# st.warning("This dashboard is still under development!")

with st.sidebar:
    st.title("Benefits of a Hydrogen Network in Europe")

    st.markdown("""
        **Fabian Neumann, Elisabeth Zeyen, Marta Victoria, Tom Brown**

        Explore trade-offs between electricity grid expansion and a new hydrogen network with repurposed gas pipelines in the European
        energy system.
    """)

    choices = {"electricity": "yes", "no-electricity": "no"}
    power_grid = st.radio(":hammer_and_wrench: Electricity network expansion", choices, format_func=lambda x: choices[x], horizontal=True)

    choices = {"hydrogen": "yes", "no-hydrogen": "no"}
    hydrogen_grid = st.radio(":hammer_and_wrench: Hydrogen network expansion", choices, format_func=lambda x: choices[x], horizontal=True)

    display = st.radio(":tv: View", ["Scenario comparison", "Spatial configurations", "System operation", "Sankey of energy flows", "Sankey of carbon flows"])

    st.info("""
        :book:  [Read the paper](http://arxiv.org/abs/2207.05816)

        :arrow_down: [Download the data](http://doi.org/10.5281/zenodo.6821258)

        :computer: [Inspect the code](http://github.com/fneum/spatial-sector)
    """)

    with st.expander("Details"):
        st.write("""
            All results were created using the open European energy system model PyPSA-Eur-Sec.
            The model covers all energy sectors including electricity, buildings, transport, agriculture and industry
            at high spatio-temporal resolution. The model code is available on
            [Github](http://github.com/pypsa/pypsa-eur-sec).
        """)


if display == "Scenario comparison":

    st.write(" ")
    
    st.image("data/graphical-abstract.png")

if display == "System operation":

    st.title("System Operation")

    _, col1, col_2, _, col_3, _ = st.columns([1,8,6,2,24,1])

    with col1:
        choices = {
            "total-electricity": "Total Electricity",
            "AC": "High Voltage Electricity",
            "low voltage": "Low Voltage Electricity",
            "H2": "Hydrogen",
            "gas": "Methane",
            #"methanol": "Methanol",
            "oil": "Liquid Hydrocarbons",
            "co2": "Carbon Dioxide",
            "co2 stored": "Stored Carbon Dioxide",
            "total-heat": "Total Heating",
            "urban central heat": "Urban Central Heating",
            "residential rural heat": "Residential Rural Building Heating"
        }
        carrier = st.selectbox("Carrier", choices, format_func=lambda x: choices[x])

    with col_2:
        choices = {
            "3H": "3-hourly",
            "24H": "daily",
            "168H": "weekly",
        }
        res = st.selectbox("Resolution", choices, format_func=lambda x: choices[x], index=1)

    with col_3:
        min_value = datetime.datetime(2013, 1, 1)
        max_value = datetime.datetime(2014, 1, 1)
        values = st.slider(
            'Select a range of values',
            min_value, max_value, (min_value, max_value),
            step=datetime.timedelta(hours=int(res[:-1])),
            format="D MMM, HH:mm",
            label_visibility='hidden'
        )

    df = nodal_balance(power_grid, hydrogen_grid, carrier)

    ts = df.loc[values[0]:values[1]].resample(res).mean()

    supply = ts.where(ts > 0).dropna(axis=1, how='all').fillna(0)
    consumption = -ts.where(ts < 0).dropna(axis=1, how='all').fillna(0)

    ylim = max(consumption.sum(axis=1).max(), supply.sum(axis=1).max())

    kwargs = dict(stacked=True, line_width=0, xlabel='', width=1300, height=350, hover=False, ylim=(0,ylim), legend='top')

    p_supply = supply.hvplot.area(**kwargs, color=[colors[c] for c in supply.columns], ylabel="Supply [GW]")
    p_consumption = consumption.hvplot.area(**kwargs, color=[colors[c] for c in consumption.columns], ylabel="Consumption [GW]")

    try:
        p_supply.get_dimension('Variable').label = ''
    except:
        pass
    p_consumption.get_dimension('Variable').label = ''

    st.bokeh_chart(hv.render(p_supply, backend='bokeh'), use_container_width=True)
    st.bokeh_chart(hv.render(p_consumption, backend='bokeh'), use_container_width=True)

if display == "Spatial configurations":

    df = load_report(power_grid, hydrogen_grid)

    #mapper1 = df.columns.get_level_values(0)
    #mapper2 = df.columns.get_level_values(1).map(rename_techs_tyndp)
    #df = df.groupby([mapper1, mapper2], axis=1).sum()

    translate_0 = {
        "demand": "Demand (TWh)",
        "capacity_factor": "Capacity Factors (%)",
        "biomass_potentials": "Potential (TWh)",
        "salt_caverns": "Potential (TWh)",
        "potential_used": "Used Potential (%)",
        "curtailment": "Curtailment (%)",
        "capacity": "Capacity (GW)",
        "io": "Import-Export Balance (TWh)",
        "lcoe": "Levelised Cost of Electricity (€/MWh)",
        "market_value": "Market Values (€/MWh)",
        "prices": "Market Prices (€/MWh)",
        "storage": "Storage Capacity (GWh)",
    }


    translate_1 = {
        "electricity": "Electricity",
        "AC": "Electricity",
        "H2": "Hydrogen",
        "hydrogen": "Hydrogen",
        "oil": "Liquid Hydrocarbons",
        "total": "Total",
        "gas": "Methane",
        "heat": "Heating",
        "offwind-ac": "Offshore Wind (AC)",
        "offwind-dc": "Offshore Wind (DC)",
        "onwind": "Onshore Wind",
        "PHS": "Pumped-hydro storage",
        "hydro": "Hydro Reservoir",
        "ror": "Run of River",
        "solar": "Solar PV (utility)",
        "solar rooftop": "Solar PV (rooftop)",
        "ground-sourced heat pump": "Ground-sourced Heat Pump",
        "air-sourced heat pump": "Air-sourced Heat Pump",
        "biogas": "Biogas",
        "solid biomass": "Solid Biomass",
        "nearshore": "Hydrogen Storage (cavern)",
    }

    df.rename(columns=translate_0, level=0, inplace=True)
    df.rename(columns=translate_1, level=1, inplace=True)


    st.title("Spatial Configurations")

    col1, col2, col3 = st.columns(3)

    options = [
        ("Nothing",),
        ("Potential (TWh)", "Hydrogen Storage (cavern)"),
        ("Potential (TWh)", "Biogas"),
        ("Potential (TWh)", "Solid Biomass"),
        ("Capacity (GW)", "Onshore Wind"),
        ("Capacity (GW)", "Offshore Wind (AC)"),
        ("Capacity (GW)", "Offshore Wind (DC)"),
        ("Capacity (GW)", "Solar PV (rooftop)"),
        ("Capacity (GW)", "Solar PV (utility)"),
        ("Capacity (GW)", "H2 Electrolysis"),
        ("Capacity (GW)", "Fischer-Tropsch"),
        ("Capacity (GW)", "Sabatier"),
        ("Capacity (GW)", "H2 Fuel Cell"),
        ("Capacity (GW)", "Hydro Reservoir"),
        ("Capacity (GW)", "Pumped-hydro storage"),
        ("Capacity (GW)", "Run of River"),
        ("Capacity (GW)", "battery charger"),
        ("Capacity (GW)", "battery discharger"),
        ("Capacity (GW)", "home battery charger"),
        ("Capacity (GW)", "home battery discharger"),
        ("Capacity (GW)", "Ground-sourced Heat Pump"),
        ("Capacity (GW)", "Air-sourced Heat Pump"),
        ("Capacity (GW)", "electricity distribution grid"),
        ("Demand (TWh)", "Electricity"),
        ("Demand (TWh)", "Heating"),
        ("Demand (TWh)", "Hydrogen"),
        ("Market Values (€/MWh)", "Onshore Wind"),
        ("Market Values (€/MWh)", "Offshore Wind (AC)"),
        ("Market Values (€/MWh)", "Offshore Wind (DC)"),
        ("Market Values (€/MWh)", "Solar PV (utility)"),
        ("Market Values (€/MWh)", "Solar PV (rooftop)"),
        ("Market Prices (€/MWh)", "Hydrogen"),
        ("Market Prices (€/MWh)", "Electricity"),
        ("Import-Export Balance (TWh)", "Total"),
        ("Import-Export Balance (TWh)", "Electricity"),
        ("Import-Export Balance (TWh)", "Hydrogen"),
        ("Import-Export Balance (TWh)", "Methane"),
        ("Import-Export Balance (TWh)", "Liquid Hydrocarbons"),
        ("Capacity Factors (%)", "Solar PV (rooftop)"),
        ("Capacity Factors (%)", "Solar PV (utility)"),
        ("Capacity Factors (%)", "Onshore Wind"),
        ("Capacity Factors (%)", "Offshore Wind (DC)"),
        ("Capacity Factors (%)", "Offshore Wind (AC)"),
        ("Storage Capacity (GWh)", "Hydrogen"),
        ("Storage Capacity (GWh)", "battery"),
        ("Storage Capacity (GWh)", "home battery"),
        ("Curtailment (%)", "Onshore Wind"),
        ("Curtailment (%)", "Offshore Wind (AC)"),
        ("Curtailment (%)", "Offshore Wind (DC)"),
        ("Curtailment (%)", "Solar PV (rooftop)"),
        ("Curtailment (%)", "Solar PV (utility)"),
    ]

    with col1:
        sel_r = st.selectbox(
            "Regions", options,
            help='Choose which data should be shown in choropleth.',
            format_func=parse_spatial_options
        )  
    
    with col2:
        network_options = [
            ("Nothing",),
            ("Hydrogen Network", "Total Capacity (GW)"),
            ("Hydrogen Network", "Retrofitted Capacity (GW)"),
            ("Hydrogen Network", "New Capacity (GW)"),
            ("Electricity Network", "Total Capacity (GW)"),
            ("Electricity Network", "Reinforcement (GW)"),
            ("Electricity Network", "Original Capacity (GW)"),
        ]
        sel_n = st.selectbox(
            "Network", network_options, 
            help='Choose which network data should be displayed.',
            format_func=parse_spatial_options
        )

    with col3:
        sel_b = st.selectbox(
            "Nodes", options,
            help='Choose which data should be shown on nodes.',
            format_func=parse_spatial_options
        )

    gdf = load_regions()

    if sel_r[0] == 'Nothing':

        kwargs = dict(
            color='white',
            line_color='grey',
            alpha=0.7,
        )
    
    else:
        # st.write(df.columns.levels[1])
        # st.write(df.columns.levels[0])

        col = " - ".join(sel_r)
        gdf[col] = df[sel_r].reindex(gdf.index)

        if "heat" in col.lower():
            cmap = "Reds"
        elif "solar" in col.lower():
            cmap = "Oranges"
        elif "hydrogen" in col.lower():
            cmap = "Purples"
        elif "bio" in col.lower() or "battery" in col.lower():
            cmap = "Greens"
        elif "Import-Export" in col:
            cmap = "PiYG_r"
        else: 
            cmap = 'Blues'

        gdf["Region"] = gdf.index
        kwargs = dict(
            alpha=0.7,
            c=col,
            cmap=cmap,
            hover_cols=['Region', col],
            clabel=col,
        )

    plot = gdf.hvplot(
        geo=True,
        height=750,
        tiles="OSM",
        **kwargs
    ).opts(
        xaxis=None,
        yaxis=None,
        active_tools=['pan', 'wheel_zoom'],
    )

    if not sel_n[0] == 'Nothing':

        if sel_n[0] == "Hydrogen Network":
            G, pos = make_hydrogen_graph()
        else:
            G, pos = make_electricity_graph()

        col = " - ".join(sel_b)
        nx.set_node_attributes(G, df[sel_b], "test")

        network_plot = hvnx.draw(
            G,
            pos=pos,
            responsive=True,
            geo=True,
            node_size=5,
            node_color='k',
            edge_color=sel_n[1],
            inspection_policy="edges",
            edge_width=hv.dim(sel_n[1]) / 2,
        ).opts(
            active_tools=['pan', 'wheel_zoom']
        )

        plot *= network_plot

    if not sel_b[0] == "Nothing":

        points = gdf.copy()
        points.geometry = points.representative_point()

        col = " - ".join(sel_b)
        points[col] = df[sel_b].reindex(gdf.index)

        node_plot = points.hvplot(
            geo=True,
            hover_cols=['Region', col],
            s=points[col]*10,
            c="#555555"
        ).opts(
            xaxis=None,
            yaxis=None,
            active_tools=['pan', 'wheel_zoom'],
        )

        plot *= node_plot

    st.bokeh_chart(hv.render(plot, backend='bokeh'), use_container_width=True)

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

