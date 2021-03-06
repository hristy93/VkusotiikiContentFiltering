import csv
import xlsxwriter

#from VkusotiikiContentFiltering import prepare_data, get_recipes_names


def create_user_likes_file():
    """Create file with recipes ids, recipes titles and default values for likes."""
    data = prepare_data().get('data')
    print()

    recipes_names = get_recipes_names(data)
    print('Add', len(recipes_names), 'recipes')

    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook('recipes.xlsx')
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})

    # Write some simple text.
    worksheet.write_row(0, 0, range(len(recipes_names)), bold)
    worksheet.write_row(1, 0, recipes_names, bold)

    for i in range(100):
        worksheet.write_row(i + 2, 0, [0] * len(recipes_names))

    workbook.close()

    print('File created!')


def read_user_likes_data(filename):
    with open(filename, 'r') as data_doc:
        d = csv.reader(data_doc, delimiter=',')
        res = []
        for index, item in enumerate(d):
            if not all(map(lambda x: x == '0', item)) and index > 1:
                res.append(tuple(map(int, item)))
        return res


if __name__ == '__main__':
    create_file()
    read_data('recipes.csv')
