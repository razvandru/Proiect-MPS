import sys 
import xlrd


if __name__ == "__main__":
    workbook = xlrd.open_workbook('mps.dataset.xlsx')
    worksheet = workbook.sheet_by_index(0)
    for i in range(0,13):
        print(worksheet.cell(0, i).value)



