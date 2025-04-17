import openpyxl
from openpyxl.styles import Border, Side, Alignment, Font
import pandas as pd

def write_dataframes_to_excel(filename, *dataframes, titles=None):
    """
    将不同的dataframe输入到同一个sheet表的指定单元格，每个dataframe之间空一行，
    并添加边框。可以为每个dataframe添加标题。

    Args:
        filename (str): Excel文件名。
        *dataframes (pd.DataFrame): 要写入的pandas DataFrames。
        titles (list of str, optional): 每个dataframe对应的标题。
    """
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    current_row = 1

    # 定义边框样式
    thin = Side(border_style="thin", color="000000")
    border = Border(top=thin, left=thin, right=thin, bottom=thin)
    center_align = Alignment(horizontal='center', vertical='center')
    bold_font = Font(bold=True)

    for i, df in enumerate(dataframes):
        # 如果有标题，写标题
        if titles and i < len(titles) and titles[i]:
            title = titles[i]
            end_col = len(df.columns)
            title_cell = sheet.cell(row=current_row, column=1, value=title)
            title_cell.font = bold_font
            title_cell.alignment = center_align

            # 合并标题单元格
            if end_col > 1:
                sheet.merge_cells(start_row=current_row, start_column=1,
                                  end_row=current_row, end_column=end_col)
            current_row += 1  # 写完标题后往下移一行

        start_row = current_row

        # 写列名
        for col_index, column in enumerate(df.columns):
            cell = sheet.cell(row=current_row, column=col_index + 1, value=column)
            cell.border = border
            cell.font = bold_font
            cell.alignment = center_align

        # 写数据
        for row_index, row_data in df.iterrows():
            current_row += 1
            for col_index, value in enumerate(row_data):
                cell = sheet.cell(row=current_row, column=col_index + 1, value=value)
                cell.border = border

        # 给整个DataFrame加边框
        end_row = current_row
        end_col = len(df.columns)
        for row in range(start_row, end_row + 1):
            for col in range(1, end_col + 1):
                cell = sheet.cell(row=row, column=col)
                cell.border = border

        current_row += 2  # 留空一行，再接下一个DataFrame

    workbook.save(filename)

if __name__ == '__main__':
    # 示例 DataFrame
    df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6, 7], 'B': [8, 9, 10], 'C': [11, 12, 13]})
    df3 = pd.DataFrame({'X': [14], 'Y': [15]})

    # 示例标题
    titles = ['表一：示例数据1', '表二：示例数据2', '表三：示例数据3']

    # 写入 Excel
    excel_filename = 'output_with_titles.xlsx'
    write_dataframes_to_excel(excel_filename, df1, df2, df3, titles=titles)

    print(f"DataFrames已成功写入到 {excel_filename}")
