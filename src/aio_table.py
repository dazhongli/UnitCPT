import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dash_table, html, dcc, callback


class AIOTable:
    def __init__(self, id, title, df: pd.DataFrame = None, save_folder=None):
        self.id = id
        self.df = df if df is not None else pd.DataFrame()
        self.save_folder = save_folder

        self.card = dbc.Card(
            [
                dbc.CardHeader(title),
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                dbc.Button(
                                    "Add Row", id=f"{self.id}-add-row", color="light", className="mr-1"),
                                dcc.Store(id=f"{self.id}-data",
                                          data=self.df.to_dict("records")),
                                dash_table.DataTable(
                                    id=f"{self.id}-table",
                                    columns=[{"name": col, "id": col}
                                             for col in self.df.columns],
                                    data=self.df.to_dict("records"),
                                    editable=True,
                                    row_deletable=True,
                                    export_format="csv",
                                    export_headers="display",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "minWidth": 95,
                                        "maxWidth": 95,
                                        "width": 95,
                                    },
                                ),
                            ]
                        )
                    ]
                ),
            ]
        )

        self.callbacks()

    def callbacks(self):
        @callback(
            Output(f"{self.id}-table", "data"),
            [Input(f"{self.id}-add-row", "n_clicks")],
            [State(f"{self.id}-table", "data"),
             State(f"{self.id}-table", "selected_rows"),
             State(f"{self.id}-table", "columns")])
        def update_table(add_row, data, selected_rows, columns):
            if not any([add_row]):
                # return the original data if none of the buttons were clicked
                return data
            # create a copy of the original data to modify
            df = pd.DataFrame(data, columns=[c["name"] for c in columns])
            if add_row:
                df = df.append(pd.Series(dtype="object"), ignore_index=True)
            if self.save_folder is not None:
                df.to_csv(os.path.join(self.save_folder,
                          f"{self.id}.csv"), index=False)

            # return the modified data
            return df.to_dict("records")
