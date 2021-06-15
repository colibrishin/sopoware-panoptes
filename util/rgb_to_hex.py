def rgb_to_hex(rgb: list):
    code = []
    for i in rgb:
        
        for j in i:
            if j < 0 or j > 255:
                assert 'Color code out of range'
        
        code.append('#%02x%02x%02x' % i)
    return code