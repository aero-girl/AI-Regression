import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Generate dummy data for demand forecasting
def create_demand_forecasting_data(num_entries=100):
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=num_entries, freq='D')
    categories = np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Home & Garden'], size=num_entries)
    sales = np.random.randint(100, 1000, size=num_entries)
    weather = np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy'], size=num_entries)
    promotion = np.random.choice([True, False], size=num_entries)
    
    return pd.DataFrame({
        'Date': dates,
        'Day of Week': dates.dayofweek,
        'Category': categories,
        'Sales': sales,
        'Weather': weather,
        'Promotion': promotion
    })

# Prepare data for model training
data = create_demand_forecasting_data()
X = data[['Day of Week', 'Category', 'Weather', 'Promotion']]
y = data['Sales']

# One-hot encode categorical variables and create a pipeline with a linear regression model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Category', 'Weather', 'Promotion'])
    ], remainder='passthrough')

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app layout
st.title('📈 Interactive Store Demand Forecasting with Sales Prediction')

st.header('📊 Dummy Dataset Overview')
st.write(data.head())

# Interactive Sales Over Time Visualization
st.header('📉 Interactive Sales Over Time')
selected_category = st.selectbox('🏷️ Select Category', ['All'] + list(data['Category'].unique()))
filtered_data = data if selected_category == 'All' else data[data['Category'] == selected_category]
st.line_chart(filtered_data[['Date', 'Sales']].set_index('Date'))

# Sales Prediction Model
st.header('🔮 Sales Prediction Model')
st.write(f'📈 Model Mean Squared Error: {mse:.2f}')
st.write('👇 Input parameters to predict sales:')

# User inputs for prediction
days_to_predict = st.slider('📅 Number of days to predict', 1, 30, 7)
category = st.selectbox('🛍️ Category for Prediction', ['Electronics', 'Clothing', 'Groceries', 'Home & Garden'])
weather = st.selectbox('☀️ Weather for Prediction', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
promotion = st.selectbox('🎉 Is there a Promotion?', [True, False])

# Predict and plot
if st.button('🚀 Generate Predictions'):
    future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
    prediction_data = pd.DataFrame({
        'Day of Week': future_dates.dayofweek,
        'Category': [category] * days_to_predict,
        'Weather': [weather] * days_to_predict,
        'Promotion': [promotion] * days_to_predict
    })
    predicted_sales = model.predict(prediction_data)
    prediction_results = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predicted_sales})
    st.bar_chart(prediction_results.set_index('Date')['Predicted Sales'])

    
st.header('📝 Generate Prediction Report')

# Button to generate report
if st.button('Generate Report'):
    # Assemble the data for the report
    report_data = prediction_results.to_dict('records')

    # Prepare the prompt for the OpenAI model
    prompt = f"Create a detailed report based on the following sales predictions: {report_data}"

    # Call OpenAI's API to generate the report (this is a mock-up)
    try:
        openai_secret_key = os.getenv('OPENAI_SECRET_KEY')
        openai.api_key = openai_secret_key

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=150
        )

        generated_report = response.choices[0].text.strip()
        st.write(generated_report)

    except Exception as e:
        st.error("Error generating report: Ensure OpenAI API key is correctly set.")
