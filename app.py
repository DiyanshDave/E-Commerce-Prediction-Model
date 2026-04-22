import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score


# LOAD DATA
df = pd.read_csv("cleaned_superstore.csv")


# DATE PARSING
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce', dayfirst=True)

df = df.dropna(subset=['Order Date'])


# FEATURE ENGINEERING
df['Order Month'] = df['Order Date'].dt.month
df['Profit_Class'] = df['Profit'].apply(lambda x: 1 if x > 0 else 0)


# ENCODING
le = LabelEncoder()
for col in ['Category', 'Region', 'Segment']:
    df[col] = le.fit_transform(df[col])


# FEATURES
features = ['Sales', 'Quantity', 'Discount', 'Category', 'Region', 'Order Month']
X = df[features]

y_reg = df['Profit']
y_clf = df['Profit_Class']


# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# TRAIN TEST SPLIT
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)


# REGRESSION
reg_model = LinearRegression().fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)
mse = mean_squared_error(y_test_reg, y_pred_reg)


# CLASSIFICATION
clf_model = LogisticRegression(max_iter=1000).fit(X_train, y_train_clf)
y_pred_clf = clf_model.predict(X_test)
accuracy = accuracy_score(y_test_clf, y_pred_clf)


# KNN
knn_data = df[['Sales','Profit','Quantity']]
knn_scaler = StandardScaler()
knn_scaled = knn_scaler.fit_transform(knn_data)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(knn_scaled)


# CLUSTERING
customer_data = df.groupby('Customer ID')[['Sales','Profit','Quantity']].sum()
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data)


# UI
st.title("E-Commerce ML System")

st.write("### Model Performance")
st.write(f"Regression MSE: {round(mse,2)}")
st.write(f"Classification Accuracy: {round(accuracy,2)}")

st.sidebar.header("Enter Details")

sales = st.sidebar.number_input("Sales", value=500.0)
quantity = st.sidebar.number_input("Quantity", value=2)

discount_percent = st.sidebar.slider("Discount", 0, 100, 10)
discount = discount_percent / 100

category_list = ['Furniture', 'Office Supplies', 'Technology']
region_list = ['Central', 'East', 'South', 'West']

category_name = st.sidebar.selectbox("Category", category_list)
region_name = st.sidebar.selectbox("Region", region_list)

month = st.sidebar.selectbox("Month", list(range(1, 13)))

category_map = {'Furniture':0, 'Office Supplies':1, 'Technology':2}
region_map = {'Central':0, 'East':1, 'South':2, 'West':3}

category = category_map[category_name]
region = region_map[region_name]

input_data = np.array([[sales, quantity, discount, category, region, month]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    profit = reg_model.predict(input_scaled)[0]
    cls = clf_model.predict(input_scaled)[0]

    st.subheader("Prediction")
    st.write("Predicted Profit:", round(profit,2))
    st.write("Profit Type:", "High" if cls==1 else "Low")

    input_knn = knn_scaler.transform([[sales, max(profit, 0), quantity]])
    distances, indices = knn.kneighbors(input_knn)
    st.subheader("Similar Products")
    st.write(df.iloc[indices[0]][['Product Name','Sales','Profit']])

st.subheader("Customer Segmentation Sample")
st.write(customer_data.head())
