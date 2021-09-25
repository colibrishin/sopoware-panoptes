import numpy as np

def get_full_probability(mask, n_classes: int):
    probability = np.zeros((n_classes), dtype=np.int32)
    updates = np.ones_like(mask, dtype=np.int32)

    np.add.at(probability, mask, updates)
    
    return probability / (mask.shape[0] * mask.shape[1])

def write_probability_table_xml(probability_table, reference):
    start_tag = '<xml><probability_table>'
    body = ''
    end_tag = '</probability_table></xml>'
    section_start = '<item>'
    section_end = '</item>'
    #square = '&#x25A0'

    for n, i in enumerate(zip(probability_table, reference)):
        p_t, ref = i
        name = '<name>' + ref[0] + '</name>'
        color = '<color>' + ref[1] + '</color>'
        probability = '<probability>' + str(p_t) + '</probability>'
        body = body + section_start + name + color + probability + section_end

    output = start_tag + body + end_tag

    with open('./probability.xml', 'w') as f:
        f.write(output)
    
