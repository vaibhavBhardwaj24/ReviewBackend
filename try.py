import json
import pandas as pd

# Load the CSV file
data = pd.read_csv('overAll.csv')

# Select distinct values from the 'brand' and 'model' columns and drop duplicates
distinct_values = data[['model', 'brand']].drop_duplicates()

# Filter for ZTE phones
zte_phones = data[data['model'].str.lower() == 'galaxy s23 ultra']

# Count the number of reviews for each model
zte_reviews_count = zte_phones.groupby('model').size().reset_index(name='review_count')

# Sort by 'review_count' in ascending order
zte_reviews_count_sorted = zte_reviews_count.sort_values(by='review_count', ascending=True)

# Convert the result to a dictionary
zte_reviews_dict = zte_reviews_count_sorted.to_dict(orient='records')

# Print the resulting dictionary in JSON format
json_output = json.dumps(zte_reviews_dict, indent=4)
print(json_output)
