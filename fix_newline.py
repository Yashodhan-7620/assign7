path = 'src/train.py' 
data = open(path, 'rb').read() 
if not data.endswith(b'\n'): 
    open(path, 'ab').write(b'\n') 
    print('Fixed: newline added') 
else: 
    print('Already fine') 
