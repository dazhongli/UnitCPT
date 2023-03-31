import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html


class AIOCard(dbc.Card):
    def __init__(self, title, label_input_pairs, update_id=None, store_id=None, *args, **kwargs):
        super().__init__(*args, style={"fontSize": "12pt"}, **kwargs)
        # Store the update ID and store ID as attributes
        self.title = title.replace(" ", "-").lower()
        self.update_id = update_id if update_id is not None else f'{self.title}-update'
        self.store_id = store_id if store_id is not None else f'{self.title}-store'

        # Create the card header with the title and link to toggle the collapse
        self.header = dbc.CardHeader(
            html.A(
                f"{title}",
                href="#",
                id=f'{self.title}-title',
                style={"textDecoration": "none", "color": "inherit",
                       "textAlign": "left", "width": "100%"}
            )
        )
        # Create the card body with the label-input pairs and update button
        form_children = []
        row_children = []
        for i, item in enumerate(label_input_pairs):
            label = item[0]
            input_component = item[1]
            property = item[2]
            input_id = f"{label.lower().replace(' ',' ')}-input"
            col_label = dbc.Col(
                html.Label(label),
                width=6,
                style={"textAlign": "left", "marginTop": "10px"}
            )
            col_input = dbc.Col(
                input_component(id=input_id, **property),
                width=6,
                style={"marginTop": "10px"}
            )
            row_children.extend([col_label, col_input])
            if (i+1) % 2 == 0 or i == len(label_input_pairs) - 1:
                form_children.append(dbc.Row(row_children))
                row_children = []
        form_children.append(
            dbc.Button(
                "Update",
                id=self.update_id,
                color='light',
                style={"width": "100%", "marginTop": "20px"}
            )
        )
        form_children.append(
            dcc.Store(
                id=self.store_id,
                data={}
            )
        )
        self.body = dbc.Collapse(
            dbc.CardBody(form_children),
            id=f"{self.title}-collapse",
            is_open=True,
        )

        # Add the header and body to the card
        self.children = [self.header, self.body]
        self.callbacks()

    def callbacks(self):
        @callback(Output(f"{self.title}-collapse", "is_open"),
                  [Input(f"{self.title}-title", "n_clicks")],
                  [State(f"{self.title}-collapse", "is_open")])
        def toggle_collapse(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

        @callback(Output(self.store_id, 'data'),
                  Input(self.update_id, 'n_clicks')
                  )
        def store_input(n_clicks):
            if n_clicks > 0:
                input_data = {}
                for item in self.body.children[:-2]:
                    if isinstance(item, dbc.Row):
                        label = item.children[0].children
                        input_id = item.children[1].children.id
                        input_value = item.children[1].children.value
                        input_data[label] = {
                            'id': input_id, 'value': input_value}
                return input_data
            else:
                return {}
