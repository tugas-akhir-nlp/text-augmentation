def list_from_file(filename):
    with open(filename) as file:
        temp = file.readlines()
        
    for i in range(len(temp)):
        temp[i] = temp[i].strip()
        
    return temp
