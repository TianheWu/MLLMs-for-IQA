import yaml



def read_yaml(path):
    with open(path, 'r') as file:
        data = file.read()
        result = yaml.load(data, Loader=yaml.FullLoader)
        return result


# print(read_yaml("./settings.yaml"))
    
print([] + ["a"])