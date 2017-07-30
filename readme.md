### Data Maining Task

#### Simple statictics and Semantics Analysis was preformed on Yelp Dataset for Gilbert City.

**Details about exercise is provided in the pdf included in exercise folder.**




#### Script
- - -
##### `json_to_csv_converter`

The script runs in Python 2. The `json_to_csv_converter` script converts the JSON format files of the Yelp Dataset Challenge to CSV format. The downloaded JSON files of the Yelp Dataset Challenge need to be kept at `/data/json`. The CSV files will be generated at `/data/csv`.

*Usage : * `python2 json_to_csv_converter.py`

Note: While running the scripts please be patient as it takes some time convert especially the review and user files.
- - -
### `extract_city`
The script runs in Python 3. It creates a subset of the dataset for the specified city in `/data/city/<city name>`

*Usage : * `python extract_city.py "city_name"`
