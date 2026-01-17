


def read_parameter_file(file_path): 
    # Read the file into a dictionary (skips empty lines and comments starting with #)
    with open(file_path, 'r') as file:
        data = {}
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                key, value = line.split('=', 1)  # Split on first '=' only
                data[key.strip()] = value.strip()
    
    return data
    
def compare_snaps(file1, file2, dict1_name='Dict1', dict2_name='Dict2'):

    dict1 = read_parameter_file(file1)
    dict2 = read_parameter_file(file2)

    keys_to_skip = ['LoadBalancingCycleSkip', 'SubcycleNumber', 'StopTime', 'InitialTime', 
                    'CurrentTimeIdentifier', 'grackle_data_file', 'MovieTimestepCounter', 'TimeLastDataDump',
                    'InitialCycleNumber', 'DataDumpNumber'] #trivially different parameters, dont report them 
    # Get sets of keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    # Keys only in dict1
    only_in1 = keys1 - keys2
    if only_in1:
        print(f"Keys only in {dict1_name}: {only_in1}")
        for key in only_in1:
            print(f"  {key}: {dict1[key]}")
    
    print()

    # Keys only in dict2
    only_in2 = keys2 - keys1
    if only_in2:
        print(f"Keys only in {dict2_name}: {only_in2}")
        for key in only_in2:
            print(f"  {key}: {dict2[key]}")
    
    print()

    # Common keys with different values
    common_keys = keys1 & keys2
    differing_values = []
    for key in common_keys:
        if dict1[key] != dict2[key]:
            differing_values.append(key)
    
    if differing_values:
        print(f"Common keys with different values:")
        for key in differing_values:
            if (key not in keys_to_skip): 
                print() 
                print(f"  {key}: ")
                print(f"         {dict1[key]} = {file1}") #{file2}={dict2[key]}")
                print(f"         {dict2[key]} = {file2}") #{file2}={dict2[key]}")


    # Summary
    if not only_in1 and not only_in2 and not differing_values:
        print("No differences found! The dictionaries are identical.")
    else:
        print(f"\nSummary: {len(only_in1)} keys only in {dict1_name}, {len(only_in2)} only in {dict2_name}, {len(differing_values)} differing values.")
