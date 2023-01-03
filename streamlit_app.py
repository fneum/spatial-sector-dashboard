import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from matplotlib.colors import to_rgba
from contextlib import suppress
from bokeh.models import HoverTool  
import geopandas as gpd
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import datetime
import hvplot.pandas

from helpers import prepare_colors, rename_techs_tyndp, get_cmap

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

    fn = f"data/{power_grid}/{hydrogen_grid}/balance-ts-{carrier}.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)

    df = df.groupby(df.columns.map(rename_techs_tyndp), axis=1).sum()

    df = df.loc[:, ~df.columns.isin(["H2 pipeline", "transmission lines"])]

    missing = df.columns.difference(preferred_order)
    order = preferred_order.intersection(df.columns).append(missing)
    df = df.loc[:, order]
    
    return df

@st.cache
def load_report(power_grid, hydrogen_grid):
    fn1 = "data/report.csv"
    fn2 = f"data/{power_grid}/{hydrogen_grid}/report.csv"
    return pd.concat([
        pd.read_csv(fn1, index_col=0, header=[0,1]),
        pd.read_csv(fn2, index_col=0, header=[0,1])
    ], axis=1)


@st.cache(allow_output_mutation=True)
def load_regions():
    fn = "data/regions_onshore_elec_s_181.geojson"
    gdf = gpd.read_file(fn).set_index('name')
    gdf['name'] = gdf.index
    return gdf


@st.cache
def load_positions():
    buses = pd.read_csv("data/buses.csv", index_col=0)
    return pd.concat([buses.x, buses.y], axis=1).apply(tuple, axis=1).to_dict()


@st.cache(allow_output_mutation=True)
def make_electricity_graph(power_grid, hydrogen_grid):

    edges = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/edges-electricity.csv", index_col=0)

    edges["Total Capacity (GW)"] = edges.s_nom_opt.clip(lower=1e-3)
    edges["Reinforcement (GW)"] = (edges.s_nom_opt - edges.s_nom).clip(lower=1e-3)
    edges["Original Capacity (GW)"] = edges.s_nom.clip(lower=1e-3)
    edges["Maximum Capacity (GW)"] = edges.s_nom_max.clip(lower=1e-3)
    edges["Technology"] = edges.carrier
    edges["Length (km)"] = edges.length

    attr = ["Total Capacity (GW)", "Reinforcement (GW)", "Original Capacity (GW)", "Maximum Capacity (GW)", "Technology", "Length (km)"]
    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)

    pos = load_positions()

    return G, pos

@st.cache(allow_output_mutation=True)
def make_hydrogen_graph(power_grid, hydrogen_grid):

    edges = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/edges-hydrogen.csv", index_col=0)

    edges["Total Capacity (GW)"] = edges.p_nom_opt.clip(lower=1e-3)
    edges["New Capacity (GW)"] = edges.p_nom_opt_new.clip(lower=1e-3)
    edges["Retrofitted Capacity (GW)"] = edges.p_nom_opt_retro.clip(lower=1e-3)
    edges["Maximum Retrofitting (GW)"] = edges.max_retro.clip(lower=1e-3)
    edges["Length (km)"] = edges.length
    edges["Name"] = edges.index

    attr = ["Total Capacity (GW)", "New Capacity (GW)", "Retrofitted Capacity (GW)", "Maximum Retrofitting (GW)", "Length (km)", "Name"]
    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)

    pos = load_positions()

    return G, pos


def parse_spatial_options(x):
    return " - ".join(x) if x != 'Nothing' else 'Nothing'


#@st.cache
def load_summary(which):

    df = pd.read_csv(f"data/{which}.csv", header=[0,1], index_col=0)

    column_dict = {
        "1.0": "without power expansion",
        "opt": "with power grid expansion",
        "H2 grid": "with hydrogen network",
        "no H2 grid": "without hydrogen network",
    }

    df.rename(columns=column_dict, inplace=True)
    df.columns = ["\n".join(col).strip() for col in df.columns.values]

    df = df.groupby(df.index.map(rename_techs_tyndp), axis=0).sum()

    missing = df.index.difference(preferred_order)
    order = preferred_order.intersection(df.index).append(missing)
    df = df.loc[order, :]

    to_drop = df.index[df.abs().max(axis=1).fillna(0.0) < 1]
    print(to_drop)
    df.drop(to_drop, inplace=True)

    return df[df.sum().sort_values().index]


### MAIN

with open("data/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

colors = prepare_colors(config)

preferred_order = pd.Index(config['preferred_order'])

## DISPLAY

st.set_page_config(
    page_title='European Hydrogen Network',
    layout="wide"
)

style = '<style>div.block-container{padding-top:.5rem; padding-bottom:0rem; padding-right:1.2rem; padding-left:1.2rem}</style>'
st.write(style, unsafe_allow_html=True)

## SIDEBAR

with st.sidebar:
    st.title("Benefits of a Hydrogen Network in Europe")

    st.markdown("""
        **Fabian Neumann, Elisabeth Zeyen, Marta Victoria, Tom Brown**

        Explore trade-offs between electricity grid expansion and a new hydrogen
        network with repurposed gas pipelines in the European energy system.
    """)

    choices = {"electricity": "yes", "no-electricity": "no"}
    power_grid = st.radio(
        ":hammer_and_wrench: Electricity network expansion",
        choices, 
        format_func=lambda x: choices[x],
        horizontal=True
    )

    choices = {"hydrogen": "yes", "no-hydrogen": "no"}
    hydrogen_grid = st.radio(
        ":hammer_and_wrench: Hydrogen network expansion",
        choices,
        format_func=lambda x: choices[x],
        horizontal=True
    )

    pages = [
        "Scenario comparison",
        "Spatial configurations",
        "System operation",
        "Sankey of energy flows",
        "Sankey of carbon flows"
    ]
    display = st.radio("Pages", pages)

    st.info("""
        :book:  [Read the paper](http://arxiv.org/abs/2207.05816)

        :arrow_down: [Download the data](http://doi.org/10.5281/zenodo.6821258)

        :computer: [Inspect the code](http://github.com/fneum/spatial-sector)
    """)

    with st.expander("Details"):
        st.write("""
            All results were created using the open European energy system model
            PyPSA-Eur-Sec. The model covers all energy sectors including
            electricity, buildings, transport, agriculture and industry at high
            spatio-temporal resolution. The model code is available on
            [Github](http://github.com/pypsa/pypsa-eur-sec).
        """)

## PAGES

if display == "Scenario comparison":

    st.title("Scenario Comparison")

    choices = config["scenarios"]
    file = st.selectbox("View", choices, format_func=lambda x: choices[x], label_visibility='hidden')

    df = load_summary(file)

    color = [colors[c] for c in df.index]

    unit = choices[file].split(" (")[1][:-1] # ugly
    tooltips = [
        ('technology', "@Variable"),
        ('value', " ".join(['@value{0.00}', unit])),
    ]
    hover = HoverTool(tooltips=tooltips)

    plot = df.T.hvplot.bar(stacked=True, height=720, color=color, line_width=0, ylabel=choices[file]).opts(tools=[hover])

    st.bokeh_chart(hv.render(plot, backend='bokeh'), use_container_width=True)

if display == "System operation":

    st.title("System Operation")

    _, col1, col_2, _, col_3, _ = st.columns([1,8,6,2,24,1])

    with col1:
        choices = config["operation"]["carrier"]
        carrier = st.selectbox("Carrier", choices, format_func=lambda x: choices[x])

    with col_2:
        choices = config["operation"]["resolution"]
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

    ylim = max(
        consumption.sum(axis=1).max(),
        supply.sum(axis=1).max()
    )

    kwargs = dict(
        stacked=True,
        line_width=0,
        xlabel='',
        width=1300,
        height=350,
        hover=False,
        ylim=(0,ylim),
        legend='top'
    )

    color = [colors[c] for c in supply.columns]
    p_supply = supply.hvplot.area(**kwargs, color=color, ylabel="Supply [GW]")

    color = [colors[c] for c in consumption.columns]
    p_consumption = consumption.hvplot.area(**kwargs, color=color, ylabel="Consumption [GW]")

    with suppress(KeyError): p_supply.get_dimension('Variable').label = ''
    with suppress(KeyError): p_consumption.get_dimension('Variable').label = ''

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

    options = config["spatial"]["nodal_options"]
    network_options = config["spatial"]["network_options"]

    options.insert(0, ["Nothing"])
    network_options.insert(0, ["Nothing"])

    options = [tuple(o) for o in options]
    network_options = [tuple(o) for o in network_options]

    _, col1, col2, col3, _ = st.columns([1,30,30,30,1])

    with col1:
        r_sel = st.selectbox(
            "Regions", options,
            help='Choose which data should be shown in choropleth.',
            format_func=parse_spatial_options
        )

    with col2:
        n_sel = st.selectbox(
            "Network", network_options,
            help='Choose which network data should be displayed.',
            format_func=parse_spatial_options
        )

    with col3:
        b_sel = st.selectbox(
            "Nodes", options,
            help='Choose which data should be shown on nodes.',
            format_func=parse_spatial_options
        )

    gdf = load_regions()

    opts = dict(
        xaxis=None,
        yaxis=None,
        active_tools=['pan', 'wheel_zoom']
    )

    if r_sel[0] == 'Nothing':

        kwargs = dict(
            color='white',
            line_color='grey',
            alpha=0.7,
        )
    
    else:

        col = " - ".join(r_sel)
        gdf[col] = df[r_sel].reindex(gdf.index)
        gdf["Region"] = gdf.index

        c = col.lower()
        cmap = get_cmap(c)

        kwargs = dict(
            alpha=0.7,
            c=col,
            cmap=cmap,
            hover_cols=['Region', col],
            clabel=col,
        )

    plot = gdf.hvplot(
        geo=True,
        height=720,
        tiles=config["tiles"],
        **kwargs
    ).opts(**opts)

    if n_sel[0] == "Hydrogen Network" and not hydrogen_grid == 'no-hydrogen':

        H, posH = make_hydrogen_graph(power_grid, hydrogen_grid)

        scale = pd.Series(nx.get_edge_attributes(H, n_sel[1])).max() / 10

        network_plot = hvnx.draw(
            H,
            pos=posH,
            responsive=True,
            geo=True,
            node_size=5,
            node_color='k',
            edge_color=n_sel[1],
            inspection_policy="edges",
            edge_width=hv.dim(n_sel[1]) / scale,
        ).opts(**opts)

        plot *= network_plot

    elif n_sel[0] == "Electricity Network":

        E, posE = make_electricity_graph(power_grid, hydrogen_grid)

        network_plot = hvnx.draw(
            E,
            pos=posE,
            responsive=True,
            geo=True,
            node_size=5,
            node_color='k',
            edge_color=n_sel[1],
            inspection_policy="edges",
            edge_width=hv.dim(n_sel[1]) / 3,
        ).opts(**opts)

        plot *= network_plot

    elif n_sel[0] == "Hydrogen Network" and hydrogen_grid == 'no-hydrogen':
        st.warning("Asked to plot hydrogen network in scenario without hydrogen network. Skipping request.")

    if not b_sel[0] == "Nothing":

        points = gdf.copy()
        coords = pd.read_csv("data/buses.csv", index_col=0)
        points.geometry = gpd.points_from_xy(coords.x, coords.y, crs=4326)

        col = " - ".join(b_sel)
        points[col] = df[b_sel].reindex(gdf.index)

        marker_size = points[col] / points[col].max() * 300

        node_plot = points.hvplot(
            geo=True,
            hover_cols=['Region', col],
            s=marker_size,
            c="#454545",
            alpha=0.7
        ).opts(**opts)

        plot *= node_plot

    st.bokeh_chart(hv.render(plot, backend='bokeh'), use_container_width=True)

if display == "Sankey of carbon flows":

    st.title("Carbon Sankes")

    df = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/sankey-carbon.csv")

    st.plotly_chart(plot_carbon_sankey(df), use_container_width=True)


if display == "Sankey of energy flows":

    st.title("Energy Sankes")

    df = pd.read_csv(f"data/{power_grid}/{hydrogen_grid}/sankey.csv")

    st.plotly_chart(plot_sankey(df), use_container_width=True)

