import numpy as np

def get_full_percentage(mask, n_classes: int):
    percentage = np.zeros((n_classes), dtype=np.int32)
    updates = np.ones_like(mask, dtype=np.int32)

    np.add.at(percentage, mask, updates)
    
    return percentage / (mask.shape[0] * mask.shape[1])

def write_percentage_table_xml(percentage_table, labels: list, hex_colors: list):
    start_tag = '<xml><percentage_table>'
    body = ''
    end_tag = '</percentage_table></xml>'
    section_start = '<item>'
    section_end = '</item>'

    for n,  in enumerate(zip(percentage_table, labels, hex_colors)):
        percentage, label, hex_color = i
        name = '<name>' + label + '</name>'
        color = '<color>' + hex_color + '</color>'
        percentage = '<percentage>' + str(percent) + '</percentage>'
        body = body + section_start + name + color + percentage + section_end

    output = start_tag + body + end_tag

    with open('./percentage.xml', 'w') as f:
        f.write(output)
    
