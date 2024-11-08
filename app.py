import os
import io
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import boto3
from dotenv import load_dotenv  # Import dotenv
from functools import lru_cache

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# ============================
# AWS S3 Configuration and Model Download
# ============================

# Retrieve environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')  # Default region if not set
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_MODEL_KEY = os.getenv('S3_MODEL_KEY')

def download_model_from_s3():
    """
    Downloads the machine learning model from Amazon S3 and loads it.
    """
    # Initialize S3 client with credentials
    s3 = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    try:
        # Fetch the model from S3
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_MODEL_KEY)
        model_data = obj['Body'].read()
        model = joblib.load(io.BytesIO(model_data))
        print("Model loaded successfully from S3.")
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise e

@lru_cache(maxsize=1)
def get_model():
    """
    Caches the loaded model to prevent redundant downloads.
    """
    return download_model_from_s3()

# Load the model once and cache it
try:
    model = get_model()
except Exception as e:
    model = None  # Handle gracefully in the app if model fails to load

# ============================
# Initialize the Dash app with Bootstrap for better styling
# ============================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# ============================
# Data Preparation
# ============================

if model is not None:
    # Load trending categories prepared during training
    trending_categories = pd.read_csv('trending_categories.csv')

    # Load the original melted data for visualization
    df_melted = pd.read_csv("final_advanced_spending_data.csv")
    spending_categories = ['Shopping', 'Entertainment', 'Food', 'Fashion', 'Healthcare', 'Transportation', 'Utilities']

    df_melted = df_melted.melt(
        id_vars=['postcode', 'month', 'latitude', 'longitude', 'Average_Age', 'Average_Income', 'Population_Size',
                 'Employment_Rate', 'Education_Level', 'Season', 'Consumer_Confidence', 'Market_Saturation',
                 'Age_Group', 'Gender', 'Mosaic_Type', 'Total_Spend', 'Online_Spend', 'Offline_Spend',
                 'Card_Type', 'Loyalty_Score', 'Savings_Rate', 'Credit_Utilization'],
        value_vars=spending_categories,
        var_name='Category',
        value_name='Category_Spend'
    )

    # Clean the 'month' column
    df_melted['month'] = df_melted['month'].str.extract('Month_(\d+)').astype(int)

    # Sort values for feature engineering
    df_melted.sort_values(by=['postcode', 'Category', 'month'], inplace=True)

    # Feature Engineering: Create lag features
    feature_list = []
    grouped = df_melted.groupby(['postcode', 'Category'])

    for (postcode, category), group in grouped:
        group = group.sort_values('month')
        for lag in range(1, 4):
            group[f'lag_{lag}'] = group['Category_Spend'].shift(lag)
        group = group.dropna(subset=[f'lag_{lag}' for lag in range(1, 4)])
        feature_list.append(group)

    df_features = pd.concat(feature_list)

    # Add Demographic and Economic Features
    df_features = df_features[['postcode', 'month', 'Category', 'Category_Spend',
                               'lag_1', 'lag_2', 'lag_3', 'Average_Age',
                               'Average_Income', 'Population_Size', 'Employment_Rate',
                               'Consumer_Confidence', 'Market_Saturation',
                               'Savings_Rate', 'Credit_Utilization', 'Gender', 'Age_Group']]

    # Encode Categorical Variables
    df_features = pd.get_dummies(df_features, columns=['Gender', 'Age_Group'])

    # Prepare Testing Data (month == 12)
    test_data = df_features[df_features['month'] == 12].copy()

    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'Average_Age', 'Average_Income',
                    'Population_Size', 'Employment_Rate', 'Consumer_Confidence',
                    'Market_Saturation', 'Savings_Rate', 'Credit_Utilization'] + \
                    [col for col in df_features.columns if col.startswith('Gender_') or col.startswith('Age_Group_')]

    X_test = test_data[feature_cols]

    # Predict using the pre-trained model
    test_data['Predicted_Spend'] = model.predict(X_test)

    # Add back categorical information
    test_data['Category_Predicted'] = df_melted.iloc[test_data.index]['Category'].values  # Ensure correct indexing
    test_data['postcode'] = df_melted.iloc[test_data.index]['postcode'].values
    test_data['month'] = df_melted.iloc[test_data.index]['month'].values

    # Identify trending categories
    trending_categories = test_data.groupby(['postcode', 'month']).apply(lambda x: x.loc[x['Predicted_Spend'].idxmax()])
    trending_categories = trending_categories.reset_index(drop=True)

    # Merge latitude and longitude for mapping
    postcode_coords = df_melted[['postcode', 'latitude', 'longitude']].drop_duplicates()
    trending_categories = trending_categories.merge(postcode_coords, on='postcode', how='left')

    # ============================
    # Visualization Components
    # ============================

    # Initial Map Figure (will be updated via callback)
    map_fig = px.scatter_mapbox(
        trending_categories,
        lat="latitude",
        lon="longitude",
        color="Category_Predicted",
        size="Predicted_Spend",
        hover_name="postcode",
        hover_data={"Predicted_Spend": ':.2f', "Category_Predicted": True},
        zoom=5,
        height=600,
        title="Trending Categories by Postcode",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    map_fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend_title_text='Category_Predicted'
    )

else:
    print("Model is not loaded. Please check the AWS S3 configuration and credentials.")
    # Handle the scenario where the model is not loaded
    trending_categories = pd.DataFrame()
    map_fig = px.scatter_mapbox()  # Empty figure

# ============================
# Dash Layout
# ============================

app.layout = html.Div([
    html.Div([
        html.H1("Advanced Spending Insights Dashboard", className="header-title")
    ], className="header"),

    html.Div([
        # Left Column: Map
        html.Div([
            dcc.Graph(
                id='map-figure',
                figure=map_fig,
                style={'height': '100%', 'width': '100%'}  # Ensure map takes full height and width
            )
        ], className="left-column"),

        # Right Column: Controls and Graphs
        html.Div([
            # Controls
            html.Div([
                html.Label("Select Postcode:"),
                dcc.Dropdown(
                    id='postcode-dropdown',
                    options=[{'label': pc, 'value': pc} for pc in sorted(trending_categories['postcode'].unique())] if not trending_categories.empty else [],
                    value=trending_categories['postcode'].iloc[0] if not trending_categories.empty else None,
                    clearable=False,
                    className="dropdown"
                ),
                html.Label("Select Category:"),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in trending_categories['Category_Predicted'].unique()] if not trending_categories.empty else [],
                    value=trending_categories['Category_Predicted'].iloc[0] if not trending_categories.empty else None,
                    clearable=False,
                    className="dropdown"
                ),
            ], className="controls-container"),

            # Graphs
            html.Div([
                dcc.Graph(id='spend-trend-graph', className="graph-container", style={"height": "400px"}),
                dcc.Graph(id='top-categories-graph', className="graph-container", style={"height": "400px"})
            ], className="graphs-container"),

            # Model Performance
            
        ], className="right-column"),
    ], className="content"),
], className="dashboard")

# ============================
# Callbacks for Interactivity
# ============================

@app.callback(
    Output('spend-trend-graph', 'figure'),
    Output('top-categories-graph', 'figure'),
    Input('postcode-dropdown', 'value'),
    Input('category-dropdown', 'value')
)
def update_graph(selected_postcode, selected_category):
    if model is None:
        # Return empty figures or some error indication
        empty_fig = px.line(title="Model not loaded.")
        return empty_fig, empty_fig

    # Spending Trend Graph for selected category
    filtered_df = df_melted[(df_melted['postcode'] == selected_postcode) & (df_melted['Category'] == selected_category)]
    if not filtered_df.empty:
        spend_trend_fig = px.line(filtered_df, x='month', y='Category_Spend',
                                  title=f'Spending Trend for {selected_category} in {selected_postcode}',
                                  markers=True)
    else:
        spend_trend_fig = px.line(title="No data available for the selected options.")

    # Top 3 Categories Line Plot
    top3_categories = df_melted[df_melted['postcode'] == selected_postcode].groupby('Category')['Category_Spend'].sum().nlargest(3).index
    top3_df = df_melted[(df_melted['postcode'] == selected_postcode) & (df_melted['Category'].isin(top3_categories))]
    if not top3_df.empty:
        top3_fig = px.line(top3_df, x='month', y='Category_Spend', color='Category',
                           title=f'Top 3 Categories Trend in {selected_postcode}')
    else:
        top3_fig = px.line(title="No data available for the selected options.")

    return spend_trend_fig, top3_fig

@app.callback(
    Output('map-figure', 'figure'),
    Input('postcode-dropdown', 'value')
)
def update_map(selected_postcode):
    if model is None or trending_categories.empty:
        # Return the initial empty map or some error indication
        return map_fig

    selected_data = trending_categories[trending_categories['postcode'] == selected_postcode]
    if not selected_data.empty:
        # Update map to zoom into selected postcode
        updated_map_fig = px.scatter_mapbox(
            selected_data, lat="latitude", lon="longitude", hover_name="postcode",
            hover_data={"Predicted_Spend": ':.2f', "Category_Predicted": True}, zoom=12, height=600, title=f"Trending Category for {selected_postcode}",
            mapbox_style="open-street-map"
        )
        updated_map_fig.update_traces(marker=dict(size=15, color='blue', symbol="circle"))
        updated_map_fig.update_layout(mapbox_center={'lat': selected_data.iloc[0]['latitude'], 'lon': selected_data.iloc[0]['longitude']},
                                     margin={"r": 0, "t": 50, "l": 0, "b": 0},
                                     legend_title_text='Category_Predicted')
    else:
        updated_map_fig = map_fig  # Fallback to initial map
    return updated_map_fig

# ============================
# Run the App
# ============================

if __name__ == '__main__':
    app.run_server(debug=True)
