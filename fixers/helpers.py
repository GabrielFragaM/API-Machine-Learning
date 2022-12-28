
def get_index_column(csv, columnName):
    columns = []
    for c in csv.axes[1]:
        columns.append(c)

    return columns.index(columnName)


def get_columns(csv):
    columns = []
    for c in csv.axes[1]:
        columns.append(c)

    return columns
