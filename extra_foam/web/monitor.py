"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys
import argparse

import dash
import dash_core_components as dcc
import dash_table as dt
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from ..ipc import init_redis_connection
from ..database import Metadata, MetaProxy, MonProxy


class Color:
    BKG = '#3AAFA9'
    SHADE = '#2B7A78'
    TEXT = '#17252A'
    TITLE = '#DEF2F1'
    INFO = '#FEFFFF'
    GRAPH = '#FBEEC1'


# update intervals in second
FAST_UPDATE = 1.0
SLOW_UPDATE = 2.0

app = dash.Dash(__name__)
# We use the default CSS style here:
# https://codepen.io/chriddyp/pen/bWLwgP?editors=1100
# app.css.append_css({
#     "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
# })
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

meta_proxy = MetaProxy()
mon_proxy = MonProxy()


def get_top_bar_cell(name, value):
    """Get cell for the top bar.

    :param str name: parameter name.
    :param str/int/float value: parameter value.
    """
    return html.Div(
        className="three-col",
        children=[
            html.P(className="p-top-bar", children=name),
            html.P(className="display-none", children=value),
            html.P(id=name, children=value),
        ],
    )


def get_top_bar():
    """Get Div for the top bar."""
    return [
        get_top_bar_cell("Detector", "Unknown"),
        get_top_bar_cell("Topic", "Unknown"),
        get_top_bar_cell("Train ID", '0' * 9),
    ]


def get_analysis_types():
    """Query and parse analysis types."""
    ret = []
    query = meta_proxy.get_all_analysis()
    if query is not None:
        for k, v in query.items():
            if k != 'AnalysisType.UNDEFINED' and int(v) > 0:
                ret.append({'type': k.split(".")[-1], 'count': v})
    return ret


def get_processor_params(proc=None):
    """Query and parse processor metadata."""
    if proc is None:
        return []
    query = mon_proxy.get_processor_params(proc)
    if query is None:
        return []
    return [{'param': k, 'value': v} for k, v in query.items()]


# define callback functions

@app.callback(output=[Output('Detector', 'children'),
                      Output('Topic', 'children'),
                      Output('Train ID', 'children')],
              inputs=[Input('fast_interval1', 'n_intervals')])
def update_top_bar(n_intervals):
    ret = mon_proxy.get_last_tid()

    if not ret:
        # [] or None
        tid = '0' * 9
    else:
        _, tid = ret

    sess = meta_proxy.hget_all(Metadata.SESSION)
    detector = "Unknown" if sess is None else sess['detector']
    topic = "Unknown" if sess is None else sess['topic']

    return detector, topic, tid


@app.callback(output=Output('analysis_type_table', 'data'),
              inputs=[Input('fast_interval2', 'n_intervals')])
def update_analysis_types(n_intervals):
    return get_analysis_types()


@app.callback(output=Output('processor_params_table', 'data'),
              inputs=[Input('fast_interval3', 'n_intervals')],
              state=[State('processor_dropdown', 'value')])
def update_processor_params(n_intervals, proc):
    return get_processor_params(proc)


@app.callback(output=Output('performance', 'figure'),
              inputs=[Input('slow_interval', 'n_intervals')])
def update_performance(n_intervals):
    ret = mon_proxy.get_latest_tids()
    if ret is None:
        raise dash.exceptions.PreventUpdate()

    tids = []
    freqs = []
    prev_timestamp = None
    for timestamp, tid in ret:
        tids.append(tid)
        float_timestamp = float(timestamp)
        if not freqs:
            freqs.append(0)
        else:
            freqs.append(1. / (prev_timestamp - float_timestamp))

        prev_timestamp = float_timestamp

    traces = [go.Bar(x=tids, y=freqs,
                     marker=dict(color=Color.GRAPH))]
    figure = {
        'data': traces,
        'layout': {
            'xaxis': {
                'title': 'Train ID',
            },
            'yaxis': {
                'title': 'Processing rate (Hz)',
            },
            'font': {
                'family': 'Courier New, monospace',
                'size': 16,
                'color': Color.INFO,
            },
            'margin': {
                'l': 100, 'b': 50, 't': 50, 'r': 50,
            },
            'paper_bgcolor': Color.SHADE,
            'plot_bgcolor': Color.SHADE,
        }
    }

    return figure


def get_monitor_layout():
    """define content and layout of the web page."""
    return html.Div(
        children=[
            dcc.Interval(
                id='fast_interval1', interval=FAST_UPDATE * 1000, n_intervals=0,
            ),
            dcc.Interval(
                id='fast_interval2', interval=FAST_UPDATE * 1000, n_intervals=0,
            ),
            dcc.Interval(
                id='fast_interval3', interval=FAST_UPDATE * 1000, n_intervals=0,
            ),
            dcc.Interval(
                id='slow_interval', interval=SLOW_UPDATE * 1000, n_intervals=0,
            ),
            html.Div([
                html.H4(
                    className='header-title',
                    children="EXtra-foam status monitor",
                ),
            ]),
            html.Div(
                id="top_bar",
                className="div-top-bar",
                children=get_top_bar(),
            ),
            html.Div(
                children=[dcc.Graph(
                    id='performance',
                )]
            ),
            html.Div(
                children=[
                    html.Div(
                        id='processor_list',
                        className='display-inlineblock',
                        children=[
                            dcc.Dropdown(
                                id='processor_dropdown',
                                options=[
                                    {'label': n.replace('_', ' '), 'value': n}
                                    for n in Metadata.processors
                                ]
                            ),
                            dt.DataTable(
                                id='processor_params_table',
                                columns=[{'name': 'Parameter', 'id': 'param'},
                                         {'name': 'Value', 'id': 'value'}],
                                data=get_processor_params(),
                                style_header={
                                    'color': Color.TEXT,
                                },
                                style_cell={
                                    'backgroundColor': Color.BKG,
                                    'color': Color.INFO,
                                    'fontWeight': 'bold',
                                    'fontSize': '18px',
                                    'text-align': 'left',
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        id='analysis_type',
                        className='display-inlineblock',
                        children=[
                            dt.DataTable(
                                id='analysis_type_table',
                                columns=[{'name': 'Analysis type', 'id': 'type'},
                                         {'name': 'Count', 'id': 'count'}],
                                data=get_analysis_types(),
                                style_header={
                                    'color': Color.TEXT,
                                },
                                style_cell={
                                    'backgroundColor': Color.BKG,
                                    'color': Color.INFO,
                                    'fontWeight': 'bold',
                                    'fontSize': '18px',
                                    'text-align': 'left',
                                },
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


def web_monitor():
    """Start a Flask server for the monitor.

    This function is for the command line tool: extra-foam-monitor.
    """
    ap = argparse.ArgumentParser(prog="extra-foam-monitor")

    ap.add_argument("--redis_address", help="Address of the Redis server",
                    default="127.0.0.1")
    ap.add_argument("--redis_port", help="Port of the Redis server",
                    default=6379)
    ap.add_argument("-p", "--password", help="Password of the Redis server",
                    default=None)

    args = ap.parse_args()
    redis_host = args.redis_address
    redis_port = args.redis_port
    password = args.password

    init_redis_connection(redis_host, redis_port, password=password)

    app.layout = get_monitor_layout()

    # The good practice is to use the predefined ports on the online cluster
    for port in [8050, 8051, 8052, 8053, 8054]:
        try:
            app.run_server(port=port)
            sys.exit(0)
        except OSError:
            print(f"\n--- Port {port} is already in use! ---\n")
            continue

    print("--- All the ports are in use! ---")
