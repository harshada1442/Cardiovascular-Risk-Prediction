import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data_cardiovascular_risk.csv')


# plt.scatter(data['age'], data['cigsPerDay'], c=data['TenYearCHD'])  # yellow : 1 , purple : 0
# plt.title('Age V/S cigsPerDay')
# plt.xlabel('Age -->')
# plt.ylabel('cigsPerDay -->')
# plt.legend(data['TenYearCHD'])
# plt.show()

def plot_scatters(data, xname: str, yname: str):
    plt.scatter(data[xname], data[yname], c=data['TenYearCHD'])  # yellow : 1 , purple : 0
    plt.title(f'{xname} V/S {yname}')
    plt.xlabel(f'{xname} -->')
    plt.ylabel(f'{yname} -->')
    plt.legend(data['TenYearCHD'])
    plt.show()


column_names = list(data.columns)
for i in range(len(column_names)):
    print(i, column_names[i], end="   ")
print()
x = int(input("Enter Column Index : "))
y = int(input("Enter Column Index : "))

plot_scatters(data, xname=column_names[x], yname=column_names[y])

