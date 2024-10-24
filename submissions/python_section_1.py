from typing import Dict, List

import pandas as pd

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result=[]
     for i in range(0, len(lst), n):
        # Define the current group
        group = []
        # Manually collect elements for the current group
        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])

        # Reverse the collected group manually
        for k in range(len(group) - 1, -1, -1):
            result.append(group[k])
    return lst
print(reverse_by_n_elements([1,2,3,4,5,6,7,8],3))
print(reverse_by_n_elements([1,2,3,4,5],2))
print(reverse_by_n_elements([10,20,30,40,50,60,70],4))   



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
   result = {}
    for word in lst:
        length = len(word)
        if length not in result:
            result[length] = []
        result[length].append(word)
    # Sort the dictionary by keys
    return dict(sorted(result.items()))
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))


from typing import Any,Dict
def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    def _flatten(d: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key, sep).items())
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = f"{parent_key}[{i}]"
                items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((parent_key, d))
        return dict(items)

    return _flatten(nested_dict)
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened = flatten_dict(nested_dict)
print(flattened)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(first = 0):
        # if all integers are used up
        if first == n:
            perm = nums[:]
            if perm not in output:
                output.append(perm)
        for i in range(first, n):
            # skip duplicates
            if i > first and nums[i] == nums[first]:
                continue
            # swap
            nums[first], nums[i] = nums[i], nums[first]
            # use next integers to complete the permutations
            backtrack(first + 1)
            # backtrack
            nums[first], nums[i] = nums[i], nums[first]

    output = []
    n = len(nums)
    backtrack()
    return output

print(unique_permutations([1, 1, 2]))


import re

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates

# Example usage:
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))


import polyline
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Radius of the Earth in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    pd.DataFrame:
    decoded_coords = polyline.decode(polyline_str)
    data = {
        "latitude": [coord[0] for coord in decoded_coords],
        "longitude": [coord[1] for coord in decoded_coords],
        "distance": [0]  # First distance is 0 since there's no previous point
    }

    for i in range(1, len(decoded_coords)):
        lat1, lon1 = decoded_coords[i - 1]
        lat2, lon2 = decoded_coords[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        data["distance"].append(distance)
    
    return pd.DataFrame(data)

# Example usage:
polyline_str = "your_polyline_string_here"
df = polyline_to_dataframe(polyline_str)
print(df)

   



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
     n = len(matrix)
    
    #  Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Create the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Sum all elements in the same row and column, excluding the current element
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]
    
    return final_matrix
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Generate the complete range of 24 hours for each day of the week
    complete_days = pd.date_range(start="2024-01-01 00:00:00", end="2024-01-07 23:59:59", freq='H').strftime('%Y-%m-%d %H:%M:%S')
    
    results = []

    for (id, id_2), group in df.groupby(['id', 'id_2']):
        timestamps = pd.concat([group['start'], group['end']]).dt.strftime('%Y-%m-%d %H:%M:%S').unique()
        complete_coverage = set(complete_days).issubset(set(timestamps))
        results.append(((id, id_2), complete_coverage))
    
    # Convert the results into a MultiIndex Series
    index = pd.MultiIndex.from_tuples([result[0] for result in results], names=['id', 'id_2'])
    return pd.Series([result[1] for result in results], index=index)

df = pd.read_csv('dataset-2.csv')
print(time_check(df))

   
