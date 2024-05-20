import os
import glob
import pandas as pd

directory_path="txt"
file_pattern = os.path.join(directory_path, '*')
files = glob.glob(file_pattern)

results = pd.DataFrame(columns=["test_case", "iteration_no", "num_vehicles", "total_distance", "objective"])
iteration_no = 0

for file_path in files:
    # file_name = "results_1716207963.txt"
    # file_path = f"txt/{file_name}"

    with open(file_path, "r") as file:
        lines = file.readlines()
        test_case = ""
        num_vehicles = -1
        objective_value = -1
        total_distance = -1
        iteration_no = iteration_no + 1
        for line in lines:
            line = line.strip()
            tokens = line.split()
            if tokens[0] == "Problem:" :
                # start parsing new test case results
                num_vehicles = -1
                objective_value = -1
                total_distance = -1
                test_case = tokens[4]
            elif tokens[0] == "Number" :
                num_vehicles = int(tokens[3])
                total_distance = float(tokens[7])
                objective_value = float(tokens[10])
                temp = pd.DataFrame([[test_case, iteration_no, num_vehicles, total_distance, objective_value]], columns=["test_case", "iteration_no", "num_vehicles", "total_distance", "objective"])
                results = pd.concat([results, temp], ignore_index=True)


summary = pd.DataFrame(columns=["test_case", "td_best", "td_mean", "td_std", "obj_best","obj_mean", "obj_std"])
test_cases = results["test_case"].unique()
for test_case in test_cases:
    td_min = results.loc[results["test_case"]==test_case, "total_distance"].min()
    td_mean = results.loc[results["test_case"]==test_case, "total_distance"].mean()
    td_std = results.loc[results["test_case"]==test_case, "total_distance"].std()
    obj_min = results.loc[results["test_case"]==test_case, "objective"].min()
    obj_mean = results.loc[results["test_case"]==test_case, "objective"].mean()
    obj_std = results.loc[results["test_case"]==test_case, "objective"].std()
    temp = pd.DataFrame([[test_case, td_min, td_mean, td_std, obj_min, obj_mean, obj_std]],columns=["test_case", "td_best", "td_mean", "td_std", "obj_best","obj_mean", "obj_std"])
    summary = pd.concat([summary, temp], ignore_index=True)
    
import pdb; pdb.set_trace()
