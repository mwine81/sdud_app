import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

st.set_page_config(layout='wide')

df = pd.read_parquet('sdud_2021_generic.parquet')
df = df.astype({'year': 'int', 'Quarter': 'int'}).astype({'year': 'str', 'Quarter': 'str'})

states = np.array(df.loc[df.Utilization_Type == 'MCOU','State'].unique())


def get_lowest(data, utilization):
    con = duckdb.connect()
    df = data.loc[data.Utilization_Type == utilization]
    sql = """SELECT *, min(avg_normalized_rx_price) over (partition by GPI,Quarter) """ + f" as {utilization}_lowest FROM df"
    df = con.execute(sql).df()
    df = df[['year', 'Quarter', 'GPI', f'{utilization}_lowest']]
    return df.drop_duplicates()


def combine_lowest(data):
    con = duckdb.connect()
    ffsu = get_lowest(data, 'FFSU')
    mcou = get_lowest(data, 'MCOU')

    sql = """
    select a.year,
    a.Quarter,
    a.GPI,
    MCOU_lowest,
    FFSU_lowest
    from mcou a
    left join ffsu b
    on a.year = b.year and a.Quarter = b.Quarter and a.GPI = b.GPI
    """

    df = con.execute(sql).df()
    return df


def add_min_price(data, q, year, state: str = 'OR'):
    df = data.loc[(data.State == state) & (data.Utilization_Type == 'MCOU') & (data.Quarter == q) & (data.year == year)]
    lowest = combine_lowest(data)
    con = duckdb.connect()
    sql = """
    SELECT a.*,
    FFSU_lowest,
    MCOU_lowest
     FROM df a
     left join lowest b
     on a.year = b.year and a.Quarter = b.Quarter and a.GPI = b.GPI
    """
    df = con.execute(sql).df()
    df['FFSU_delta'] = df.FFSU_lowest - df.avg_normalized_rx_price
    df['MCOU_delta'] = df.MCOU_lowest - df.avg_normalized_rx_price
    df['nadac_delta'] = df.avg_normalized_nadac_per_rx - df.avg_normalized_rx_price

    return df.get(['GPI_14_Name','avg_units_by_gpi','avg_normalized_rx_price','avg_normalized_nadac_per_rx','FFSU_delta','MCOU_delta','nadac_delta']).sort_values(by='MCOU_delta')


def state_price_data(data, q, drug, utilization):
    df = data.loc[(data.Quarter == q) & (data.GPI_14_Name == drug) & (data.Utilization_Type == utilization)]
    return df

def create_map(data, q, drug, utilization):
    df = state_price_data(data, q, drug, utilization)
    fig = px.choropleth(df, locationmode="USA-states", locations='State', color='avg_normalized_rx_price', scope="usa")

    return fig


st.title('SDUD Managed Care Pricing Application')

st.markdown('Select a state')

quarter = st.selectbox(
    'Which quarter would you like to view?',
    ['1', '2', '3', '4']
)

year = st.selectbox(
    'Which quarter would you like to view?',
    ['2021']
)

state = st.selectbox(
    'Which state would you like to view?',
    np.sort(states))

selection = add_min_price(df, state=state, year=year,q=quarter)

st.dataframe(selection)

with st.expander('Expand for explanation'):
    st.markdown("""
    - GPI's where standardized to a mean number of units per prescription based on yearly reported data
    - Each GPI number of units per prescription was multiplied by the mean price per unit from SDUD reportings
    - The NADAC was estimated per GPI utilizing a quarterly weighted unit price
    - The min price paid within each program was determined
    - the FFSU delta is the difference between the experienced MCOU payment and the lowest reported FFSU by a state (a 
    negative number indicates an overpayment)
    - the MCOU delta is the difference between the experienced MCOU payment and the lowest reported MCOU by a state (a 
    negative number indicates an overpayment)
    - the NADAC delta is the difference between the experienced MCOU payment and the weighted quarterly NADAC (a 
    negative number indicates an overpayment)
    """)


drugs = sorted(list(selection['GPI_14_Name']))

drug = st.selectbox(
    'Which drug would you like to view',
    drugs
)

def get_price_state(how,utilization,data=df,quarter=quarter,drug=drug):
    df = state_price_data(data, quarter, drug, utilization).get(['State', 'avg_normalized_rx_price'])
    if how == 'min':
        df = df.loc[df.avg_normalized_rx_price == df.avg_normalized_rx_price.min()]
    if how == 'max':
        df = df.loc[df.avg_normalized_rx_price == df.avg_normalized_rx_price.max()]
    return df['State'].iloc[0], '$' + str(round(df['avg_normalized_rx_price'].iloc[0],2))


col1, col2 = st.columns(2)
with col1:
    min_mcou_state, min_mcou_price = get_price_state('min','MCOU')
    max_mcou_state, max_mcou_price = get_price_state('max', 'MCOU')
    st.markdown('## MCOU')
    st.markdown(f'{min_mcou_state} had the lowest managed care price per prescription at {min_mcou_price}')
    st.markdown(f'{max_mcou_state} had the highest managed care price per prescription at {max_mcou_price}')
    st.plotly_chart(create_map(df,q=quarter,drug=drug, utilization='MCOU'))
with col2:
    min_ffs_state, min_ffs_price = get_price_state('min','FFSU')
    max_ffs_state, max_ffs_price = get_price_state('max', 'FFSU')
    st.markdown('## FFSU')
    st.markdown(f'{min_ffs_state} had the lowest fee for service price per prescription at {min_ffs_price}')
    st.markdown(f'{max_ffs_state} had the highest fee for service price per prescription at {max_ffs_price}')
    st.plotly_chart(create_map(df,q=quarter,drug=drug,utilization='FFSU'))

