from dash import callback, html, dash_table, dcc, Input, Output, State, MATCH
import fakeredis
import hashlib
import io
import json
import os
import pandas as pd
import plotly
import redis
import warnings
import uuid

class redis_store:
    """Save data to Redis using the hashed contents as the key.
    Serialize Pandas DataFrames as memory-efficient Parquet files.

    Otherwise, attempt to serialize the data as JSON, which may have a
    lossy conversion back to its original type. For example, numpy arrays will
    be deserialized as regular Python lists.

    Connect to Redis with the environment variable `REDIS_URL` if available.
    Otherwise, use FakeRedis, which is only suitable for development and
    will not scale across multiple processes.
    """
    if 'REDIS_URL' in os.environ:
        r =  redis.StrictRedis.from_url(os.environ["REDIS_URL"])
    else:
        warnings.warn('Using FakeRedis - Not suitable for Production Use.')
        r = fakeredis.FakeStrictRedis()

    @staticmethod
    def _hash(serialized_obj):
        return hashlib.sha512(serialized_obj).hexdigest()

    @staticmethod
    def save(value):
        if isinstance(value, pd.DataFrame):
            buffer = io.BytesIO()
            value.to_parquet(buffer, compression='gzip')
            buffer.seek(0)
            df_as_bytes = buffer.read()
            hash_key = redis_store._hash(df_as_bytes)
            type = 'pd.DataFrame'
            serialized_value = df_as_bytes
        else:
            serialized_value = json.dumps(value, cls=plotly.utils.PlotlyJSONEncoder).encode('utf-8')
            hash_key = redis_store._hash(serialized_value)
            type = 'json-serialized'

        redis_store.r.set(
            f'_dash_aio_components_value_{hash_key}',
            serialized_value
        )
        redis_store.r.set(
            f'_dash_aio_components_type_{hash_key}',
            type
        )
        return hash_key

    @staticmethod
    def load(hash_key):
        data_type = redis_store.r.get(f'_dash_aio_components_type_{hash_key}')
        serialized_value = redis_store.r.get(f'_dash_aio_components_value_{hash_key}')
        try:
            if data_type == b'pd.DataFrame':
                value = pd.read_parquet(io.BytesIO(serialized_value))
            else:
                value = json.loads(serialized_value)
        except Exception as e:
            print(e)
            print(f'ERROR LOADING {data_type - hash_key}')
            raise e
        return value


# DataTable-specific Filtering logic from https://dash.plotly.com/datatable/callbacks
_operators = [
    ['ge ', '>='],
    ['le ', '<='],
    ['lt ', '<'],
    ['gt ', '>'],
    ['ne ', '!='],
    ['eq ', '='],
    ['contains '],
    ['datestartswith ']]


def _split_filter_part(filter_part):
    for operator_type in _operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                # word _operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

class DataTableAIO(html.Div):
    class ids:
        datatable = lambda aio_id: {
            'component': 'DataTableAIO',
            'subcomponent': 'datatable',
            'aio_id': aio_id
        }
        store = lambda aio_id: {
            'component': 'DataTableAIO',
            'subcomponent': 'store',
            'aio_id': aio_id
        }
    ids = ids

    def __init__(self, df=None, aio_id=None, **datatable_props):
        """DataTableIO is an All-in-One component that is composed of a parent `html.Div`
        with a `dcc.Store` and a `dash_table.DataTable` as children.
        The dataframe filtering, paging, and sorting is performed in a built-in
        callback that uses Pandas.

        The DataFrame is stored in Redis as a Parquet file via the
        `redis_store` class. The `dcc.Store` contains the Redis key to the
        DataFrame and can be retrieved with `redis_store.get(store['df'])`
        in a separate callback.

        The underlying functions that filter, sort, and page the data are
        accessible via `filter_df`, `sort_df`, and `page_df` respectively.

        - `df` - A Pandas dataframe
        - `aio_id` - The All-in-One component ID used to generate the `dcc.Store` and `DataTable` components's dictionary IDs.
        - `**datatable_props` - Properties passed into the underlying `DataTable`
        """
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        # Infer DataTable column types from the Pandas DataFrame
        columns = []
        columns_cast_to_string = []
        for c in df.columns:
            column = {'name': c, 'id': c}
            dtype = pd.api.types.infer_dtype(df[c])
            if dtype.startswith('mixed'):
                columns_cast_to_string.append(c)
                df[c] = df[c].astype(str)

            if pd.api.types.is_numeric_dtype(df[c]):
                column['type'] = 'numeric'
            elif pd.api.types.is_string_dtype(df[c]):
                column['type'] = 'text'
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                column['type'] = 'datetime'
            else:
                columns_cast_to_string.append(c)
                df[c] = df[c].astype(str)
                column['type'] = 'text'
            columns.append(column)

        if columns_cast_to_string:
            warnings.warn(
                'Converted the following mixed-type columns to ' +
                'strings so that they can be saved in Redis or JSON: ' +
                f'{", ".join(columns_cast_to_string)}'
            )

        derived_kwargs = datatable_props.copy()

        # Store the DataFrame in Redis and the hash key in `dcc.Store`
        # Allow the user to pass in `df=` or `data=` as per `DataTable`.
        store_data = {}
        if df is None and 'data' in datatable_props:
            store_data['df'] = redis_store.save(
                pd.DataFrame(datatable_props['data'])
            )
        elif df is not None and not 'data' in datatable_props:
            store_data['df'] = redis_store.save(df)
        elif df is not None and 'data' in datatable_props:
            raise Exception('The `df` argument cannot be supplied with the data argument - it\'s ambiguous.')
        else:
            raise Exception('No data supplied. Pass in a dataframe as `df=` or a list of dictionaries as `data=`')

        # Allow the user to pass in their own columns, otherwise define our own.
        if df is not None:
            if 'columns' not in datatable_props:
                derived_kwargs['columns'] = columns

        # Allow the user to override these properties, otherwise provide defaults
        derived_kwargs['page_current'] = derived_kwargs.get('page_current', 0)
        derived_kwargs['page_size'] = derived_kwargs.get('page_size', 10)
        derived_kwargs['page_action'] = derived_kwargs.get('page_action', 'custom')
        derived_kwargs['filter_action'] = derived_kwargs.get('filter_action', 'custom')
        derived_kwargs['filter_query'] = derived_kwargs.get('filter_query', '')
        derived_kwargs['sort_action'] = derived_kwargs.get('sort_action', 'custom')
        derived_kwargs['sort_mode'] = derived_kwargs.get('sort_mode', 'multi')
        derived_kwargs['sort_by'] = derived_kwargs.get('sort_by', [])

        super().__init__([
            dcc.Store(data=store_data, id=self.ids.store(aio_id)),
            dash_table.DataTable(id=self.ids.datatable(aio_id), **derived_kwargs)
        ])

    def filter_df(df, filter_query):
        """Filter a Pandas dataframe as per the `filter_query` provided by
        the DataTable.
        """
        filtering_expressions = filter_query.split(' && ')
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = _split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these _operators match pandas series operator method names
                df = df.loc[getattr(df[col_name], operator)(filter_value)]
            elif operator == 'contains':
                df = df.loc[df[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                df = df.loc[df[col_name].str.startswith(filter_value)]
        return df

    def sort_df(df, sort_by):
        """Sort a Pandas dataframe as per the DataTable `sort_by` property.
        """
        if len(sort_by):
            df = df.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )
        return df

    def page_df(df, page_current, page_size):
        """Page a Pandas dataframe as per the DataTable `page_current`
        and `page_size` parameters.
        """
        return df.iloc[page_current * page_size: (page_current + 1) * page_size]

    @callback(
        Output(ids.datatable(MATCH), 'data'),
        Input(ids.datatable(MATCH), 'page_current'),
        Input(ids.datatable(MATCH), 'page_size'),
        Input(ids.datatable(MATCH), 'sort_by'),
        Input(ids.datatable(MATCH), 'filter_query'),
        State(ids.store(MATCH), 'data')
    )
    def filter_sort_page(page_current, page_size, sort_by, filter, store):
        df = redis_store.load(store['df'])
        df = DataTableAIO.filter_df(df, filter)
        df = DataTableAIO.sort_df(df, sort_by)
        df = DataTableAIO.page_df(df, page_current, page_size)
        return df.to_dict('records')
