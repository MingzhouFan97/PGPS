with open(f'./german.data-numeric', 'r') as f:
    aaa = f.read()
    bbb = aaa.replace('   ', ' ')
    ccc = bbb.replace('  ', ' ')
    ddd = ccc.replace('\n ', '\n')

with open(f'./german.data-processed', 'w') as f:
    f.write(ddd)