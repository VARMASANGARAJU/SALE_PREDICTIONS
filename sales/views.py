from django.shortcuts import render, redirect
import csv,io
# Create your views here.

def index(request):
    template = "sales/index.html"
    return render(request,template)


def upload_csv(request):
    template = "sales/upload.html"

    prompt = {
        'order' : 'Order of CSV should be x,y,z,m'
    }

    if request.method == "GET":
        return render(request,template,prompt)

    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a csv file')

    data_set = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(data_set)
    csv_data = []
    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        csv_data.append(column)

    print(csv_data[:10])
    context = {
        'headers' : csv_data[0],
        'all_data' : csv_data[1:]
    }
    return render(request,template,context)


def train_model(data):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn import metrics

    # loading the data
    sales_data = pd.read_csv('static/Train.csv')
    #checking the first 5 rows of the dataframe

    # mean value of "Item_Weight" column
    sales_data['Item_Weight'].mean()

    # filling the missing values in "Item_weight column" with "Mean" value
    sales_data['Item_Weight'].fillna(sales_data['Item_Weight'].mean(), inplace=True)

    # mode of "Outlet_Size" column
    sales_data['Outlet_Size'].mode()

    # filling the missing values in "Outlet_Size" column with Mode
    #Here we take Outlet_Size column & Outlet_Type column since they are correlated
    mode_of_Outlet_size = sales_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

    # sales_data.loc[miss_values, 'Outlet_Size'] = sales_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


    # Data Processing
    sales_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

    encoder = LabelEncoder()

    sales_data['Item_Identifier'] = encoder.fit_transform(sales_data['Item_Identifier'])

    sales_data['Item_Fat_Content'] = encoder.fit_transform(sales_data['Item_Fat_Content'])

    sales_data['Item_Type'] = encoder.fit_transform(sales_data['Item_Type'])

    sales_data['Outlet_Identifier'] = encoder.fit_transform(sales_data['Outlet_Identifier'])

    sales_data['Outlet_Size'] = encoder.fit_transform(sales_data['Outlet_Size'])

    sales_data['Outlet_Location_Type'] = encoder.fit_transform(sales_data['Outlet_Location_Type'])

    sales_data['Outlet_Type'] = encoder.fit_transform(sales_data['Outlet_Type'])




    #Let's have all the features in X & target in Y
    X = sales_data.drop(columns='Item_Outlet_Sales', axis=1)
    Y = sales_data['Item_Outlet_Sales']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    print(X.shape, X_train.shape, X_test.shape)

    regressor = XGBRegressor()

    #fit the model
    #Training data is in X_train and the corresponding price value is in the Y_train
    regressor.fit(X_train, Y_train)


    sales_data_prediction = regressor.predict(X_train)
    # In order to check the performance of the model we find the R squared Value
    r2_sales = metrics.r2_score(Y_train, sales_data_prediction)
    print('R Squared value = ', r2_sales)

    # prediction on test data
    data_prediction = regressor.predict(X_test)

    # R squared Value
    r2_data = metrics.r2_score(Y_test, data_prediction)

    print('R Squared value = ', r2_data)


    input_data = (156, 9.300, 0, 0.016047, 4, 249.8092, 9, 1999,1, 0, 1)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = regressor.predict(input_data_reshaped)

    context = {"final_prediction": prediction[0],
                "dataset_value" :  sales_data_prediction[0] }


    return context

def data_input(request):
    template = "sales/Predict.html"
    context = {}
    if request.method == "POST":
        data = {}
        data['item_type'] = request.POST.get('item_type')
        data['item_weight'] = request.POST.get('item_weight')
        data['item_fat_content'] = request.POST.get('item_fat_content')
        data['item_mrp'] = request.POST.get('item_mrp')
        data['item_visibility'] = request.POST.get('item_visibility')
        data['outlet_establishment_year'] = request.POST.get('outlet_establishment_year')
        data['outlet_size'] = request.POST.get('outlet_size')
        data['outlet_location_type'] = request.POST.get('outlet_location_type')
        data['outlet_type'] = request.POST.get('outlet_type')

        context = train_model(data)

    return render(request,template,context)
