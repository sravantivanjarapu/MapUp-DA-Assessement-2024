import numpy as np
import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
     locations = pd.concat([df['id'], df['id_2']]).unique()
    n = len(locations)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    for _, row in df.iterimport pandas as pd

        distance_matrix.at[row['id'], row['id_2']] = row['distance']
        distance_matrix.at[row['id_2'], row['id']] = row['distance']
    np.fill_diagonal(distance_matrix.values, 0)
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    return distance_matrix
df = pd.read_csv('dataset-2.csv')
print(calculate_distance_matrix(df))  




def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    rows = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            distance = distance_matrix.at[id_start, id_end]
              if id_start != id_end and distance < float('inf'):
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    df = pd.DataFrame(rows)
    return df
distance_matrix = calculate_distance_matrix(df)  # Assume this is the result from the previous function
df = unroll_distance_matrix(distance_matrix)
print(df)    




def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])
    reference_average = reference_distances.mean()
    lower_bound = reference_average * 0.9
    upper_bound = reference_average * 1.1
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = average_distances[(average_distances['distance'] >= lower_bound) &
                                      (average_distances['distance'] <= upper_bound)]
    df = filtered_ids.sort_values(by='distance')                                  

    return df

df = pd.read_csv('dataset-2.csv')  # Assuming this is the output from the previous function
reference_id = some_id  # Specify a valid reference ID
result = find_ids_within_ten_percentage_threshold(df, reference_id)
print(result)




def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df
df = pd.read_csv('dataset-2.csv')  
result_df = calculate_toll_rate(df)
print(result_df)



from datetime import time
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_intervals = {
        "weekday_morning": (time(0, 0), time(10, 0)),
        "weekday_day": (time(10, 0), time(18, 0)),
        "weekday_evening": (time(18, 0), time(23, 59, 59)),
        "weekend": (time(0, 0), time(23, 59, 59))
    }
     start_days = []
    end_days = []
    start_times = []
    end_times = []
    for i in range(len(df)):
        start_day_index = i % 7  # Cycle through days
        end_day_index = (i + 1) % 7 
        start_days.append(days_of_week[start_day_index])
        end_days.append(days_of_week[end_day_index])
     df['start_day'] = start_days
    df['end_day'] = end_days
    df['start_time'] = start_times
    df['end_time'] = end_times
    def get_discount_factor(row):
        day = row['start_day']
        start_time = row['start_time']
        
        if day in ["Saturday", "Sunday"]:
            return 0.7
            
        else:
            if time(0, 0) <= start_time < time(10, 0):
                return 0.8  # Morning rate
            elif time(10, 0) <= start_time < time(18, 0):
                return 1.2  # Day rate
            else:
                return 0.8  # Evening rate
    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
        df[vehicle] *= df.apply(get_discount_factor, axis=1) 

    return df
df = pd.read_csv('dataset-2.csv')  # Assuming this is the output from the previous function
result_df = calculate_time_based_toll_rates(df)
print(result_df)
