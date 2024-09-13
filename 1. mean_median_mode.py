from statistics import mode
def calculate_median(data):
    data = sorted(data)
    n=len(data)
    mid = n//2
    if n%2 == 0:
        median = (data[mid-1]+data[mid]) / 2
    else:
        median = data[mid]
    return median

def calculate_mean(data):
    mean = sum(data)/len(data)
    return mean


data_set = {10,15,20,25,30,15,20,15,19,30,15}
mean_result = calculate_mean(data_set)
print("Mean: ",mean_result)
median_result = calculate_median(data_set)
print("Median: ", median_result)
mode_result = calculate_median(data_set)
print("Mode: ", mode_result)

