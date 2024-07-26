import pandas as pd
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

def main():
    nanoindentation_data = load_data('NanoIndentation Data\Substrate_Hertzian Contact.xlsx')
    # print(nanoindentation_data.to_string())
    # print((-2)**1.5) <-- results in a complex number
    axis_limits = user_input(min(nanoindentation_data['h (Indentation Depth)']), min(nanoindentation_data['P (Applied Load)']))
    fitting_params = fit_data(nanoindentation_data[nanoindentation_data['h (Indentation Depth)'] >= 0])
    print(f'Here are the fitting parameters when fitted to P = a(h^3/2) + b \n' \
          f'where a and b are fitting parameters: \na = {fitting_params[0]} \n' \
            f'b = {fitting_params[1]}')
    text = f'P = a(h^3/2) + b \na = {round(fitting_params[0], 2)} \nb = {round(fitting_params[1], 2)}'
    print(findx(2000, fitting_params[0], fitting_params[1])) # finds x if 2000 is the load applied to the material
    fitting_xvals = np.linspace(0, findx(axis_limits[3], fitting_params[0], fitting_params[1]), 1000) # 1000 evenly spaced values between 0 and the max indentation depth


    plot_data(nanoindentation_data, fitting_params, axis_limits, fitting_xvals, text)


def user_input(xmin, ymin) -> list:
    try:
        ymax = float(input('Please enter the highest load that you would like to see on the graph: '))
        xmax = float(input('Please enter the highest indentation depth that you would like to see on the graph: '))
        return [xmin, xmax, ymin, ymax]
    except ValueError:
        print('Please enter a valid number')
        user_input()
    pass

def function(params, x) -> float:
    a, b = params
    return b + a * x**(1.5)

def findx(load, a, b) -> float:
    return ((load - b) / a)**(2/3)

def load_data(excel_file:str) -> pd.DataFrame:
    excel_data = pd.read_excel(excel_file, header=None)
    excel_data.columns = ['P (Applied Load)', 'h (Indentation Depth)']
    excel_data = excel_data.dropna()
    return excel_data

def fit_data(data:pd.DataFrame) -> list:
    model = odr.Model(function) # using 3/2 polynomial order
    data = odr.Data(x=data['h (Indentation Depth)'], y=data['P (Applied Load)'])
    odr_instance = odr.ODR(data, model, beta0=[1.0, 1.0])
    output = odr_instance.run()
    return output.beta

def plot_data(data:pd.DataFrame, fitted_data_params:list, axis_limits:list, xvals:np.array, text:str):
    plt.figure()
    plt.xlabel('h (Indentation Depth)', fontweight='bold')
    plt.ylabel('P (Applied Load)', fontweight='bold')
    plt.plot(data['h (Indentation Depth)'], data['P (Applied Load)'], 'ro', label='Raw Data')
    plt.plot(xvals, function(fitted_data_params, xvals), 'b-', label='Fitted Data')
    plt.text(0.01, 0.99, text, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontweight='bold')
    plt.axis(axis_limits)
    plt.legend(prop={'weight': 'bold'})
    plt.show()


if __name__ == '__main__':
    main()
