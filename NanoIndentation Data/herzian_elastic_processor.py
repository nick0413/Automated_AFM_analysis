import pandas as pd
from scipy import odr
import matplotlib.pyplot as plt

def main():
    nanoindentation_data = load_data('NanoIndentation Data\Substrate_Hertzian Contact.xlsx')
    # print(nanoindentation_data.to_string())
    # print((-2)**1.5)
    fitting_params = fit_data(nanoindentation_data[nanoindentation_data['h (Indentation Depth)'] >= 0])
    print(f'Here are the fitting parameters when fitted to P = a(h^3/2) + b \n' \
          f'where a and b are fitting parameters: \na = {fitting_params[0]} \n' \
            f'b = {fitting_params[1]}')
    plot_data(nanoindentation_data, fitting_params)

def function(params, x):
    a, b = params
    return b + a * x**(1.5)

def load_data(excel_file:str):
    excel_data = pd.read_excel(excel_file, header=None)
    excel_data.columns = ['P (Applied Load)', 'h (Indentation Depth)']
    excel_data = excel_data.dropna()
    return excel_data

def fit_data(data:pd.DataFrame):
    model = odr.Model(function) # using 3/2 polynomial order
    data = odr.Data(x=data['h (Indentation Depth)'], y=data['P (Applied Load)'])
    odr_instance = odr.ODR(data, model, beta0=[1.0, 1.0])
    output = odr_instance.run()
    return output.beta

def plot_data(data:pd.DataFrame, fitted_data_params:list):
    plt.figure()
    plt.xlabel('h (Indentation Depth)')
    plt.ylabel('P (Applied Load)')
    plt.plot(data['h (Indentation Depth)'], data['P (Applied Load)'], 'ro', label='Raw Data')
    filtered_data = data[data['h (Indentation Depth)'] >= 0]
    plt.plot(filtered_data['h (Indentation Depth)'], function(fitted_data_params, filtered_data['h (Indentation Depth)']), 'b-', label='Fitted Data')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
