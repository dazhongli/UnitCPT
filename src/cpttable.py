import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

class AIOCard(dbc.Card):
    def __init__(self, title, label_input_pairs, update_id=None, store_id=None, *args, **kwargs):
        super().__init__(*args, color="light", style={"fontSize": "12pt"}, **kwargs)
        
        # Store the update ID and store ID as attributes
        self.update_id = update_id
        self.store_id = store_id
        
        # Create the card header with the title
        header = dbc.CardHeader(title)
        self.children.append(header)
        
        # Create the card body with the label-input pairs and update button
        body = dbc.CardBody()
        form_group_list = []
        for label, input_component, *input_args in label_input_pairs:
            input_id = f"{label.lower().replace(' ', '-')}-input"
            form_group = dbc.FormGroup(
                [
                    dbc.Label(label, width=6),
                    dbc.Col(input_component(id=input_id, *input_args), width=6),
                ],
                row=True,
                className="mb-3",
            )
            form_group_list.append(form_group)
        form = dbc.Form(form_group_list)
        body.children.append(form)
        
        # Add the update button
        if update_id:
            button = dbc.Button("Update", color="light", id=update_id, className="mt-3")
            body.children.append(button)
        
        # Add the body to the card
        self.children.append(body)
        
        # Create the callback to update the store component
        if store_id:
            input_ids = [f"{label.lower().replace(' ', '-')}-input" for label, input_component, *input_args in label_input_pairs]
            @callback(Output(store_id, "data"), [Input(update_id, "n_clicks")], [State(input_id, "value") for input_id in input_ids])
            def update_store(n_clicks, *input_values):
                if n_clicks:
                    data = dict(zip([label for label, input_component, *input_args in label_input_pairs], input_values))
                    return data
