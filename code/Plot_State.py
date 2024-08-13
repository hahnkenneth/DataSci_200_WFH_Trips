import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data_df, region_name='default', trip_range='default', per_capita=True, figsize=(8, 5), ax=None):

    """
    Plots the number of trips per capita and the percentage of remote jobs from January 2019 to September 2023 for a specific state or for the entire U.S.

    Parameters:
    - df (DataFrame): DataFrame containing the data with specific column requirements.
    - state_name (str): State code (e.g., 'CA') for state-specific data or 'default' for nationwide data.
    - trip_range (str): Specifies the trip distance categories to include: 'default', 'low', 'medium', or 'high'.
    - per_capita (bool): If True, plots the number of trips per capita; otherwise, plots total number of trips.
    - figsize (tuple): Figure size in inches, given as (width, height).
    - ax (matplotlib axes object, optional): Axes object on which the plot will be drawn. If None, a new figure and axes are created.

    Returns:
    - None. Displays a matplotlib plot.

    Notes:
    - The 'data_df' DataFrame is expected to have the following columns:
      'Unnamed: 0', 'date', 'state', 'population_staying_at_home', 'population_not_staying_at_home', 'number_of_trips',
       'number_of_trips_<1', 'number_of_trips_1-3', 'number_of_trips_3-5', 'number_of_trips_5-10', 'number_of_trips_10-25',
       'number_of_trips_25-50', 'number_of_trips_50-100','number_of_trips_100-250', 'number_of_trips_250-500', 'number_of_trips_>=500', 
       'state_name', 'region', 'division', 'percent', 'n', 'total_population'.
    - The 'date' column should be in datetime format.
    - The 'percent' column represents the percentage of remote jobs.
    - The 'total_population' column represents the total population of the region.

    Example:
    >>> plot_data(df, 'CA', trip_range='low', per_capita=True, figsize=(10, 6))
    """
    if region_name != 'default':
        # Create region DataFrame
        region_df = data_df[data_df['state'] == region_name].copy()  # Make a copy
    else:
        region_df = data_df.copy()  # Make a copy

    if per_capita:
        # Calculate per capita columns
        trip_columns = ['number_of_trips_<1', 'number_of_trips_1-3', 'number_of_trips_3-5', 'number_of_trips_5-10',
                        'number_of_trips_10-25', 'number_of_trips_25-50', 'number_of_trips_50-100',
                        'number_of_trips_100-250', 'number_of_trips_250-500', 'number_of_trips_>=500']
        region_df.loc[:, trip_columns] = region_df[trip_columns].div(region_df['total_population'], axis=0)

    if region_name != 'default':
        drop_columns = ['Unnamed: 0', 'state_name', 'region', 'division', 'state', 'n', 'population_staying_at_home',
                        'population_not_staying_at_home', 'number_of_trips', 'total_population']
    else:
        drop_columns = ['Unnamed: 0', 'country', 'population_staying_at_home', 'population_not_staying_at_home',
                        'number_of_trips', 'total_population']

    if trip_range == 'default':
        region_df.drop(columns=drop_columns, inplace=True)
    elif trip_range == 'low':
        drop_columns.extend(['number_of_trips_10-25', 'number_of_trips_25-50', 'number_of_trips_50-100',
                              'number_of_trips_100-250', 'number_of_trips_250-500', 'number_of_trips_>=500'])
        region_df.drop(columns=drop_columns, inplace=True)
    elif trip_range == 'medium':
        drop_columns.extend(['number_of_trips_<1', 'number_of_trips_1-3', 'number_of_trips_3-5',
                              'number_of_trips_5-10', 'number_of_trips_100-250', 'number_of_trips_250-500',
                              'number_of_trips_>=500'])
        region_df.drop(columns=drop_columns, inplace=True)
    elif trip_range == 'high':
        drop_columns.extend(['number_of_trips_<1', 'number_of_trips_1-3', 'number_of_trips_3-5',
                              'number_of_trips_5-10', 'number_of_trips_10-25', 'number_of_trips_25-50',
                              'number_of_trips_50-100'])
        region_df.drop(columns=drop_columns, inplace=True)

    region_df.set_index('date', inplace=True)

    # Plot formatting
    label_dict = {'percent': "%" + ' of Remote Jobs', 'number_of_trips_<1': 'Trips Less than 1 Mile Per Capita',
                    'number_of_trips_1-3': 'Trips Between 1 - 3 Miles Per Capita',
                    'number_of_trips_3-5': 'Trips Between 3 - 5 Miles Per Capita',
                    'number_of_trips_5-10': 'Trips Between 5 - 10 Miles Per Capita',
                    'number_of_trips_10-25': 'Trips Between 10 - 25 Miles per Capita',
                    'number_of_trips_25-50': 'Trips Between 25 - 50 Miles Per Capita',
                    'number_of_trips_50-100': 'Trips Between 50 - 100 Miles Per Capita',
                    'number_of_trips_100-250': 'Trips Between 100 - 250 Miles Per Capita',
                    'number_of_trips_250-500': 'Trips Between 250 - 500 Miles Per Capita',
                    'number_of_trips_>=500': 'Trips Greater than 500 Miles Per Capita'}

    line_style_dict = {'percent': '-', 'number_of_trips_<1': '-', 'number_of_trips_1-3': '-',
                        'number_of_trips_3-5': '-', 'number_of_trips_5-10': '-', 'number_of_trips_10-25': '-',
                        'number_of_trips_25-50': '-', 'number_of_trips_50-100': '-',
                        'number_of_trips_100-250': '-', 'number_of_trips_250-500': '-',
                        'number_of_trips_>=500': '-'}

    color_dict = {'percent': 'orange', 'number_of_trips_<1': '#1f77b4', 'number_of_trips_1-3': '#aec7e8',
                    'number_of_trips_3-5': '#6baed6', 'number_of_trips_5-10': '#3182bd',
                    'number_of_trips_10-25': '#1f77b4', 'number_of_trips_25-50': '#aec7e8',
                    'number_of_trips_50-100': '#6baed6', 'number_of_trips_100-250': '#aec7e8',
                    'number_of_trips_250-500': '#6baed6', 'number_of_trips_>=500': '#3182bd'}
    
    marker_dict = {'percent': 'o', 'number_of_trips_<1': 'x', 'number_of_trips_1-3': 'D',
                    'number_of_trips_3-5': 'o', 'number_of_trips_5-10': '^', 'number_of_trips_10-25': 'x',
                    'number_of_trips_25-50': 'D', 'number_of_trips_50-100': 'o',
                    'number_of_trips_100-250': 'x', 'number_of_trips_250-500': 'D',
                    'number_of_trips_>=500': 'o'}
    
    # Create the plot
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax

    for var in region_df.columns:
        if var == 'percent':  # For the 'percent' column, plot on the secondary axis
            ax2 = ax1.twinx()
            ax2.plot(region_df.index, region_df[var], label=label_dict[var], color=color_dict[var],
                    linestyle=line_style_dict[var], marker=marker_dict[var], linewidth=2, markersize=6)
            ax2.set_ylabel(label_dict[var])
        else:
            ax1.plot(region_df.index, region_df[var], label=label_dict[var], color=color_dict[var],
                    linestyle=line_style_dict[var], marker=marker_dict[var], markersize=6)

    # Add vertical lines for COVID dates
    ax1.axvline(x=pd.to_datetime('2020-04-01'), color='gray', linestyle='--', label='Start of COVID-19 Outbreak')
    ax1.axvline(x=pd.to_datetime('2021-02-01'), color='gray', linestyle=':', label='End of COVID-19 Outbreak')

    # Set primary y-axis label
    if per_capita:
        ax1.set_ylabel('Number of Trips Per Capita')
    else:
        ax1.set_ylabel('Number of Trips')

    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)

    # Set title
    if ax is None:
        fig.tight_layout()
        if per_capita and region_name != 'default':
            plt.title(f'Number of Trips per Capita and % Remote Jobs in {region_name}')
        elif region_name != 'default':
            plt.title(f'Number of Trips and % Remote Jobs in {region_name}')
        elif per_capita and region_name == 'default':
            plt.title(f'Number of Trips per Capita and % Remote Jobs in the U.S')
        else:
            plt.title(f'Number of Trips and % Remote Jobs in the U.S')

        # Collect handles and labels for legend
        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Move 'percent' label to the beginning of the legend
        percent_index = labels.index("% of Remote Jobs")
        handles.insert(0, handles.pop(percent_index))
        labels.insert(0, labels.pop(percent_index))

        # Create a single legend
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.7))
